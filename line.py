from linebot import LineBotApi
from linebot.models import ImageSendMessage
from linebot.models import TextSendMessage 
import requests
from datetime import datetime
import os 

    
    
def linebot(file_name,message):

    access_token = "oQQGWuZZ8HlOTXRlkZ3/Lg8BiyavBI7910RJGnIfLcKOgYdNb/IvrVGnAGkbw5wR3qgm4G/lF7aoUGK7PdCvCkjQ2Hd4+Sjx+Nw/HmE7fik5URt4xNHbk3vTtx5Bsa343laA2QvfQ1yZmhj9lGNyXwdB04t89/1O/w1cDnyilFU="#作成したbotのアクセストークン
    line_api = LineBotApi(access_token)
    tdy = str(datetime.now())
   

    #lineapiでテキストを送信
    line_api.broadcast(TextSendMessage(message))
    line_api.broadcast(TextSendMessage(tdy))
    image_message = ImageSendMessage(original_content_url = 'https://drive.google.com/uc?id='+ file_name, preview_image_url='https://drive.google.com/uc?id='+file_name) #サーバーの画像のパス
    line_api.broadcast(image_message)

    print("test")
   
    return 