import socket
import threading
import SocketServer
import time
import sys

reqs = []
num = int(sys.argv[1])
finished = 0

class ThreadedTCPRequestHandler(SocketServer.StreamRequestHandler):

    def handle(self):
        self.data = self.rfile.readline().strip()
        cur_thread = threading.current_thread()
        cport = int(self.data)
        response = "{0}: {1} | {2} | {3}".format(cur_thread.name, self.data,num,self.client_address[0]);
        mynum = len(reqs) / 2
        reqs.append(self.client_address[0])
        reqs.append(cport)
        print "Recv: {0} of {1}\n".format(len(reqs) / 2, num);
        while len(reqs) != num*2:
            time.sleep(0.0001)
        rep = ",".join(str(x) for x in reqs)
        print rep
        self.request.sendall(str(num) + "," + str(mynum) + "," + rep + "\n")
        global finished
        finished = finished + 1
        if finished == num:
            self.server.shutdown()

class ThreadedTCPServer(SocketServer.ThreadingMixIn, SocketServer.TCPServer):
    allow_reuse_address=True

if __name__ == "__main__":
    # Port 0 means to select an arbitrary unused port
    HOST, PORT = "", 9999

    
    server = ThreadedTCPServer((HOST, PORT), ThreadedTCPRequestHandler)
    ip, port = server.server_address

    # Start a thread with the server -- that thread will then start one
    # more thread for each request
    server_thread = threading.Thread(target=server.serve_forever)
    # Exit the server thread when the main thread terminates
    server_thread.daemon = True
    server_thread.start()
    print "Server loop running in thread: {0} host: {1} port: {2}".format(server_thread.name, ip, port)

    server.serve_forever(0.05)
