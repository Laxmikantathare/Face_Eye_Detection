from flask import Flask,Response,render_template
import cv2
app = Flask(__name__)
camera = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success,frame =camera.read()
        if success:
            detector=cv2.CascadeClassifier('Haarcascade/haarcascade_frontalface_default.xml')
            eye=cv2.CascadeClassifier('Haarcascade/haarcascade_eye.xml')
            face=detector.detectMultiScale(frame,1.4,9)
            grey=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            for(x,y,w,h) in face:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                roi_grey = grey[x:x+w,y:y+h]
                roi_color = frame[x:x+w,y:y+h]
                eyes=eye.detectMultiScale(roi_grey,1.1,3)
                for(ex,ey,ew,eh) in eyes:
                    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,200),2)

            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()
            yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        else:
            break

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    app.run(debug=True)

