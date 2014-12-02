from itty import *

#===============================================================================
# @get('/')
# def index(request):
#     return 'Hello World!'
#===============================================================================

@post('/')
def test_post(request):
    return 'hello world ronak'
    #return "'foo' is: %s" % request.POST.get('foo', 'not specified')

run_itty()
