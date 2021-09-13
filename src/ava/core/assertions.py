def assert_object_type(object, required_type, context=''):
    if type(required_type) not in {list, tuple, set}:
        required_type = set([required_type])

    if len(context) != 0:
        context += ': '

    msg = context + 'Object type must one of {}. Actual type: {}'
    msg = msg.format(', '.join([t.__name__ for t in required_type]), type(object).__name__)
    assert type(object) in required_type, msg

def assert_equal_shape(tuple_a, tuple_b, context=''):
    if len(context) != 0:
        context += ': '
    assert tuple_a == tuple_b, context + 'shapes are not equal. Actual sizes: ' + str(tuple_a) + ' and ' + str(tuple_b)
