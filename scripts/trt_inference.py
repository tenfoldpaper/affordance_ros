#!/usr/bin/python3
import cv2
import tensorrt as trt
import numpy as np
import os
import time
import pycuda.driver as cuda
import pycuda.autoinit
import torch
#from ava.core.visualize.color_overlay import color_overlay
from ava.core.transformations.resize import tensor_resize

# ros imports
import rospy
import rospkg
from sensor_msgs.msg import Image
from yolov5_ros.msg import DetectionMsg, DetectionArray
from cv_bridge import CvBridge, CvBridgeError
import message_filters
import queue 

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

class TrtModel:
    
    def __init__(self,engine_path,max_batch_size=1,dtype=np.float32,\
        rgb_topic = '', 
        depth_topic = '',
        rgb_camera = '', 
        depth_camera = '',
        detection_topic = '',
        min_threshold = 50,
        save_img = False,
        publish = False):
        
        package_path = rospkg.RosPack().get_path('affordance_ros')
        affordance_path = os.path.join(package_path, 'data/ade/affordances12.csv')

        affordances12 = open(affordance_path).read().split('\n')[0].split(';')[2:]
        
        # Define desired affordances to detect
        self.AFFORDANCES = ['grasp'] # no place_on because we don't need the table to be high res
        self.AFF_INDICES = [affordances12.index(aff) for aff in self.AFFORDANCES]
        self.INTENSITIES = [5, 5, 5]
        self.batch_size = 1
        self.min_threshold = min_threshold # need to make this part adjustable
        self.msg_queue = queue.Queue(maxsize=1) # we always want the freshest frame to process

        self.engine_path = engine_path
        self.dtype = dtype
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.load_engine(self.runtime, self.engine_path)
        self.max_batch_size = max_batch_size
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()
        self.context = self.engine.create_execution_context()
        self.running_total = 0.
        self.running_counter = 0
        self.rgb_topic = rgb_topic
        self.depth_topic = depth_topic
        self.publish = publish
        self.save_img = save_img
        self.bridge = CvBridge()
        # NODE specific setup
        self.img_subscriber = message_filters.Subscriber(self.rgb_topic, Image, queue_size=5, buff_size=2**24)
        self.depth_subscriber = message_filters.Subscriber(self.depth_topic, Image, queue_size=5, buff_size=2**24)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.img_subscriber, self.depth_subscriber], queue_size=1, slop=0.05)
        self.ts.registerCallback(self.detector_callback)
        
        if(self.publish):
            rospy.loginfo("Publish mode. initializing publisher.")
            self.detection_publisher = rospy.Publisher(detection_topic, DetectionMsg, queue_size=1)
        
        rospy.loginfo("Initialised TRT affordance network with the following parameters:\n" + \
            f"Model               : {engine_path}\n" +
            f"Chosen affordances  : {self.AFFORDANCES}\n" +
            f"RGB topic           : {rgb_topic}\n" +
            f"Depth topic         : {depth_topic}\n" +
            f"RGB camera topic    : {rgb_camera}\n" +
            f"Depth camera topic  : {depth_camera}\n" +
            f"Detection topic     : {detection_topic}\n" +
            f"Save image          : {save_img}\n" +
            f"Publish             : {publish}\n")
        pass
            
        
    @staticmethod
    def load_engine(trt_runtime, engine_path):
        trt.init_libnvinfer_plugins(None, "")             
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        engine = trt_runtime.deserialize_cuda_engine(engine_data)
        return engine
    
    def allocate_buffers(self):
        
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.max_batch_size
            host_mem = cuda.pagelocked_empty(size, self.dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        
        return inputs, outputs, bindings, stream
       
            
    def __call__(self, x:np.ndarray,batch_size=2):
        
        x = x.astype(self.dtype)
        np.copyto(self.inputs[0].host,x.ravel())
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp.device, inp.host, self.stream)
        self.context.execute_async(batch_size=batch_size, bindings=self.bindings, stream_handle=self.stream.handle)
        
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out.host, out.device, self.stream) 
        
        self.stream.synchronize()
        return [out.host.reshape(batch_size,-1) for out in self.outputs]
    
    def detector_callback(self, rgb_data, depth_data):
        self.msg_queue.put((rgb_data, depth_data))

    def process_frame(self):
        # initialise the message and save the time
        msg = DetectionMsg()
        rgb_data, depth_data = self.msg_queue.get(block=True)
        msg.image_time = rgb_data.header.stamp
        start = time.time()
        # first, convert the obtained Image data to a cv2-type format, which is just a numpy array.
        img = self.bridge.imgmsg_to_cv2(rgb_data, "bgr8")
        
        # Image processing

        # Test it with a dummy img
        #####        
        img = tensor_resize(img, (480, 999), interpret_as_max_bound=True)
        img_t = torch.from_numpy(img.transpose([2, 0, 1]).astype('float32'))
        img_t = img_t.unsqueeze(0)
        # Inference step
        result = self.__call__(img_t.numpy(), self.batch_size) #Inference takes about 0.23 
        
        # Combine the relevant affordance outputs 
        result_2 = result[0].reshape((12, 480, 640))
        affseg_img = np.zeros_like(result_2[0,:,:])
        
        # Bitwise or to combine the affordances into a single mask
        for i in range(0,len(self.AFF_INDICES)):
            #cv2.imwrite(f'./result_1_{i}.png', result_1[:, :, i])
            result = result_2[self.AFF_INDICES[i],:,:]/np.max(result_2[self.AFF_INDICES[i],:,:]) * 255
            result = cv2.threshold(result, self.min_threshold, 255, cv2.THRESH_BINARY)[1] 
            affseg_img = cv2.bitwise_or(affseg_img, result)
        # Bitwise or takes about 0.05 sec
        
        
        affseg_img = affseg_img.astype(np.uint8) # Gotta set it to uint8 to stop findCountours from complaining; right now it's in float.

        if(self.save_img):
            affseg_img_copy = affseg_img.copy()
            affseg_img_copy = cv2.cvtColor(affseg_img_copy, cv2.COLOR_GRAY2BGR)
            
        # Find the bounding box
        cnts = cv2.findContours(affseg_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Contour extraction takes about 0.005 sec
         
        
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        detection_count = 0
        xywh_tuple = []
        for c in cnts:
            x,y,w,h = cv2.boundingRect(c) # normalize. This is also xy origin, wh size, not xy center, wh size
            # so on top of normalizing, we need to recenter it
            x_orig, y_orig, w_orig, h_orig = x, y, w, h
            x += w/2
            y += h/2
            
            x /= 640
            w /= 640
            y /= 480
            h /= 480
            if(x < 0 or y < 0 or w < 0 or h < 0):
                rospy.loginfo(x, w, y, h)
            if(w < 15/640 and h < 15/640):
                continue
            if(w < 10/640 or h < 10/480): # we don't want TOO narrow objects; they are useless.
                continue 
            if self.save_img:
                affseg_img_copy = cv2.rectangle(affseg_img_copy, (x_orig, y_orig), (x_orig + w_orig, y_orig + h_orig), (255,0,0), 2)
            # append to tuple to be assigned later
            xywh_tuple.append([x,y,w,h])
            detection_count += 1
        
        # This drawing process takes another 0.005 sec, but it will be discarded. 
        # Need to check how long packing everything into a ROS message will take though, but I think I can get a 3~4 FPS performance out of this.
        # Create the message to be sent
        msg.rgb_image = rgb_data
        depth_data.encoding = "mono16"
        msg.depth_image = depth_data
        msg.detection_count = detection_count
        msg.detection_array = []
        
        msg.segmentation_image = self.bridge.cv2_to_imgmsg(np.array(affseg_img), 'mono8')
        msg.segmented = True
        if(self.save_img):
            cv2.imwrite('../bien_thesis/affseg_raw.jpg', img)
            cv2.imwrite('../bien_thesis/affseg_mask.jpg', affseg_img)
            cv2.imwrite('../bien_thesis/affseg_contour.jpg', affseg_img_copy)
        # Also need to attach segmentation data
        for i in range(0, len(xywh_tuple)):
            tempArr = DetectionArray()
            xywh = xywh_tuple[i]
            tempArr.detection_info = np.asarray((0, *xywh)).astype(np.float) # 0 is supposed to be a class integer, but it's not needed. just need to fill up the space
            msg.detection_array.append(tempArr)
        end = time.time()
        self.running_total += end - start
        self.running_counter += 1
        if(self.publish):
            self.detection_publisher.publish(msg)
            rospy.loginfo("Inference took {:.3f} sec, avg {:.3f} sec over {} samples. Detected {} blobs. Publishing.".format(end - start, self.running_total/self.running_counter, self.running_counter, detection_count))
        else:
            rospy.loginfo("Inference took {:.3f} sec, avg {:.3f} sec over {} samples. Detected {} blobs. Not publishing.".format(end - start, self.running_total/self.running_counter, self.running_counter, detection_count))
if __name__ == "__main__":

    rospy.init_node("affordance_node")
    # Get parameters from roslaunch
    rgb_topic = rospy.get_param("~source_topic")    
    depth_topic = rospy.get_param("~depth_topic")
    rgb_camera = rospy.get_param("~camera_topic")
    depth_camera = rospy.get_param("~depth_camera_topic")
    detection_topic = rospy.get_param("~detection_topic")
    custom_model_path = rospy.get_param("~affordance_model_path")
    min_threshold = int(rospy.get_param("~min_threshold"))
    publish = rospy.get_param("~publish")
    save_img = rospy.get_param("~save_img")
    
    package_path = rospkg.RosPack().get_path('affordance_ros')
    model_path = os.path.join(package_path, 'data/trained_res18.trt')
    
    if(custom_model_path):
        custom_model_path = os.path.join(custom_model_path, 'data/trained_res18.trt')
        try:
            model = TrtModel(custom_model_path, \
                rgb_topic = rgb_topic, 
                depth_topic = depth_topic, 
                rgb_camera = rgb_camera, 
                depth_camera = depth_camera, 
                detection_topic = detection_topic,
                min_threshold = min_threshold,
                save_img = save_img, 
                publish = publish)
        except:
            model = TrtModel(model_path, \
                rgb_topic = rgb_topic, 
                depth_topic = depth_topic, 
                rgb_camera = rgb_camera, 
                depth_camera = depth_camera,
                detection_topic = detection_topic,
                min_threshold = min_threshold,
                save_img = save_img, 
                publish = publish)
    sleep_time = rospy.Rate(10)
    while not rospy.is_shutdown():
        try:
            model.process_frame()
            sleep_time.sleep()
        except KeyboardInterrupt:
            print("Shutting down")
            break

