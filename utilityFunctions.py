
def printMessage(msg):
    print '[ORION]:', msg
    
class HandleError:
    @staticmethod
    def exit(msg):
        print "ERROR:", msg
        exit(1)