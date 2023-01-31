from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tflite_runtime.interpreter as tflite
import cv2


app = FastAPI()

# MODEL = tf.keras.models.load_model(".\model.h5")
interpreter = tflite.Interpreter("model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

CLASS_NAMES = ['agal_spot','Anthracnose','left_blinght','normal','sooty_mold']

discriptions_1 =[
                 "สาเหตุ : เกิดจาก สาหร่าย Cephaleuros virescens Kunze",
                 "สาเหตุ : เกิดจากเชื้อรา Colletotrichum zibethinum Sacc." ,
                 "สาเหตุ : เกิดจากเชื้อรา Phomopsis sp. " ,
                 "ปกติดี",
                 "สาเหตุ : เกิดจากเชื้อรา Capnodium sp. Mont. ", 
                ]
discriptions_2 =[
                 "การแพร่กระจายของโรค : แพร่ระบาดไปกับลมและพายุฝน เข้าทำลายในสภาพอากาศที่มีความชื้นสูง นอกจากนี้ น้ำก็เป็นพาหะนำสปอร์ไปสู่ต้นอื่นได้เช่นเดียวกัน",
                 "การแพร่กระจายของโรค : ลมและฝนพัดพาโรคจากใบและกิ่งสู่ดอก",
                 "การแพร่กระจายของโรค : จะแพร่ระบาดไปโดยลม และ ฝน และ จากเนื้อเยื่อใบที่แห้งและหล่นตกคางอยู่มีใต้โคนต้น เชื้อสามารถเขาทำลายได้ทั้งใบอ่อนและใบแก",
                 "ปกติดี",
                 "การแพร่กระจายของโรค : สปอร์ของเชื้อราสาเหตุโรคจะฟุ้งกระจายอยู่ในอากาศ (Air-borne fungi) เมื่อลมพัดพาสปอร์ไปตกบริเวณ ที่มีอาหารสําหรับการเจริญเติบโต ของเชื้อราๆ จะสร้างเส้นใยไมซีเลียมและเจริญ อยู่ บนผิวใบ" , 
                ]
discriptions_3 = [
                  "อาการของโรค : ใบแก่ของทุเรียนจะมีจุดฟูเขียวแกมเหลืองของสาหร่าย เกิดกระจายบนใบทุเรียน จุดจะพัฒนาและขยายออกและเปลียนเป็นสีเหลืองแกมส้มและในช่วงนี้สาหร่ายจะขยายพันธ์แพร่ระบาดต่อไป",
                  "อาการของโรค : บนใบแผลเป็นจุดวงสีน้ำตาลแดงซ้อนกัน" ,
                  "อาการของโรค : เชื้อจะเข้าทำลายที่บริเวณปลายใบไม้และขอบใบไม้ก่อน เกิดอาการปลายใบแห้ง และ ขอบใบแห้ง ที่จุดเชื้อสาเหตุเข้าทำลาย เนื้อใบส่วนนั้นจะแห้งเป็นสีน้ำตาลแดงในระยะแรก และต่อมาจะเปลี่ยนเป็นสีขาวอมเทา และเชื้อจะเจริญพัฒนาทำความเสียหายกับใบทุเรียน ขยายขนาดของพื้นที่เนื้อใบแห้งออกไปเรื่อยๆ เนื้อใบส่วนที่แห้งสีขาวอมเทามีการสร้าง ส่วนขยายพันธ์เป็นเม็ดสีดำกระจัดกระจายเต็มพื้นที่",
                  "ปกติดี",
                  "อาการของโรค : จุดราสีดำบนผิวใบ" , 
                 ]
discriptions_4 = [
                  "การป้องกันโรค : 1) ถ้ามีการระบาดของโรคมากแล้ว ควรผสมสารป้องกันกำจัดโรคพืชประเภทดูดซึมที่มีประสิทธภาพด้วย เช่นสาร กลุ่มรหัส 1( เบนโนมิล คาร์เบนดาซิม ไธอะเบนดาโซล ไทโอฟาเนทเมทิล) สารกลุ่มรหัส 3 ( ไตรฟอรีน โพรคลอราช ไดฟิโนโคนาโซล อีพ๊อกซีโคนาโซล เฮกซาโคนา โซลไมโคลบิวทานิล โพรพิโคนาโซล ทีบูโคนาโซล และ เตตราโคนาโซล เป็นต้น ) และสารกลุ่มที่มีรหัส 11 ( อะซ๊อกซีสโตรบิน ไพราโคลสโตรบิน ครึโซซิมเมทิล และ ไตรฟล๊อกซีสโตรบิน เป็นต้น) และ สลับด้วยสารประเภทสัมผัส เช่น สารกลุ่มคอปเปอร์ แมนโคเซ็บ โพรพิเนป และ คลอโรทาโลนิล เป็นต้น",
                  "การป้องกันโรค : 1) เมื่อสำรวจพบอาการของโรค ควรพ่นสารป้องกันกำจัดโรคพืชเป็นระยะๆ เช่น คาร์เบนดาซิม, โพรคลาราซ, ไตรฟลอกซีสโตรบิน, โพรพิเนบ หรือ พ่นด้วยเชื้อราไตรโคเดอร์ม่า อัตราส่วน 1 กก./น้ำ 200 ลิตร พ่นซ้ำ 2-3 ครั้ง ห่างกัน 3-5 วัน หรืออัตราส่วนที่ระบุตามฉลาก",
                  "การป้องกันโรค : 1) สารป้องกันกำจัดโรคพืชที่มีประสิทธิภาพ เช่นสาร กลุ่มรหัส 1( เบนโนมิล คาร์เบนดาซิม ไธอะเบนดาโซล ไทโอฟาเนทเมทิล) สารกลุ่มรหัส 3 ( ไตรฟอรีน โพรคลอราช ไดฟิโนโคนาโซล อีพ๊อกซีโคนาโซล เฮกซาโคนา โซลไมโคลบิวทานิล โพรพิโคนาโซล ทีบูโคนาโซล และ เตตราโคนาโซล เป็นต้น ) และสารกลุ่มที่มีรหัส 11 ( อะซ๊อกซีสโตรบิน ไพราโคลสโตรบิน ครึโซซิมเมทิล และ ไตรฟล๊อกซีสโตรบิน เป็นต้น) และ สลับด้วยสารประเภทสัมผัส เช่น สารกลุ่มคอปเปอร์ แมนโคเซ็บ โพรพิเนป และ คลอโรทาโลนิล 2) ตัดแต่งกิ่งที่เป็นโรค เพื่อลดปริมาณเชื้อสาเหตุในแปลงปลูก แล้วพ่นสารป้องกันกำจัดโรคพืชที่มีประสิทธิภาพ", 
                  "ปกติดี",
                  "การป้องกันโรค : 1) มีการระบาดของโรคแล้ว ควรผสมสารป้องกันกำจัดโรคพืชที่มีประสิทธิภาพรวมไปด้วย เช่นสารกลุ่มรหัส 1 (เบนโนมิล คาร์เบนดาซิม ไธอะเบนดาโซล ไทโอฟาเนทเมทิล) สารกลุ่มรหัส 3 (ไตรฟอรีน โพรคลอราช ไดฟิโนโคนาโซล อีพ๊อกซีโคนาโซล เฮกซาโคนาโซล ไมโคลบิวทานิล โพรพิโคนาโซล ทีบูโคนาโซล และ เตตรานาโซล เป็นต้น) และสารกลุ่มรหัส 11 ( อะซ๊อกซีสโตรบิน ไพราโคลสโตรบิน ครีโซซิมเมทิล และ ไตรฟล๊อกซีสโตรบิน เป็นต้น) และ สลับด้วยสารประเภทสัมผัสเช่น สารกลุ่มคอปเปอร์ แมนโคเซ็บ โพรพิเนป และ คลอโรทาโลนิล เป็นต้น" , 
                 ]

CLASS_DICT = {'agal_spot' : 0 ,
              'Anthracnose' : 1,
              'left_blinght' : 2,
              'normal' : 3,
              'sooty_mold': 4 
             }

def how(label):
    return discriptions_1[CLASS_DICT[label]] , discriptions_2[CLASS_DICT[label]] , discriptions_3[CLASS_DICT[label]] , discriptions_4[CLASS_DICT[label]]

def get_prediction(interpreter, input_data):
    input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_details = interpreter.get_output_details()
    return interpreter.get_tensor(output_details[0]['index'])

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    imgori = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
    img = cv2.resize(imgori ,(128,128))
    rimg = np.array(img)
    rimg = rimg.astype('float32')
    rimg /= 255
    rimg = np.reshape(rimg ,(1,128,128,3))

    predictions = get_prediction(interpreter, rimg)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0]) 
    dis1,dis2,dis3,dis4 = how(predicted_class)
    return {
        'class': predicted_class,
        'confidence': float(confidence),
        'comment_1' : dis1 ,
        'comment_2' : dis2 ,
        'comment_3' : dis3 ,
        'comment_4' : dis4 ,
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)