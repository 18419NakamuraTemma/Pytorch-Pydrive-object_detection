import os
import datetime
import time
import schedule
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import detect
import glob
import shutil

gauth = GoogleAuth()
gauth.LocalWebserverAuth()


drive = GoogleDrive(gauth)
imgs_files = glob.glob("imgs/*")
output_files = glob.glob("det/*")

drive_tempfolder_id = '1uSYBfzJQNoiHGp5p7ADgyYa09aPTKQXD' # 画像の場所
drive_savefolder_id = '1Q4qvSIx9EW4TWF9SDnakvZBs5p5SIjvB' # 画像の保存場所（ドライブ）
drive_outputfolder_id = '1cGCMeLKEPRK26JAIBdmltKMbqtTCFoO5' # 物体検知で出力されたのファイル保存場所
#save_folder = os.getcwd() + "imgs" # 画像の保存場所（ローカル）
save_folder = '/home/sakamoto/Pytorch-Pydrive-object_detection/imgs' # 画像の保存場所（ローカル）
det_save_folder = '/home/sakamoto/Pytorch-Pydrive-object_detection/det'#物体検知で出力されたのファイル保存場所(ローカル)

def change_folder(folder_id):
    for f in drive.ListFile({'q': 'mimeType = "image/jpeg" and "{}" in parents'.format(drive_tempfolder_id)}).GetList():
        f['parents'] = [{'id': folder_id}]
        f.Upload()
        print(f['parents'])

def upload_file(file_path,folder_id):
        f = drive.CreateFile({"parents": [{'mimeType':'image/jpeg',"id": folder_id}]})
        f.SetContentFile(file_path)
        f['title'] = os.path.basename(file_path)
        f.Upload()
    


def download_recursively(save_folder, drive_folder_id):
    # 保存先フォルダがなければ作成
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    max_results = 100
    query = "'{}' in parents and trashed=false".format(drive_folder_id)

    for file_list in drive.ListFile({'q': query, 'maxResults': max_results}):
        for file in file_list:
            # mimeTypeでフォルダか判別
            if file['mimeType'] == 'application/vnd.google-apps.folder':
                download_recursively(os.path.join(save_folder, file['title']), file['id'])
            else:
                file.GetContentFile(os.path.join(save_folder, file['title']))


def main(): 
#while True:    #テストのときに使用
        print('0')
        download_recursively(save_folder, drive_tempfolder_id)
        detect.main()
        
        
        change_folder(drive_savefolder_id)
        print('1')
        
        output_files = glob.glob("det/*")
        for file in output_files :
            print('3')
            upload_file(file,drive_outputfolder_id)
            

        print('-------------------------------')
        target_dir = 'imgs'
        shutil.rmtree(target_dir)
        os.mkdir(target_dir)

        target_dir = 'det'
        shutil.rmtree(target_dir)
        os.mkdir(target_dir)
        time.sleep(30)
##テストのときはコメントアウト
schedule.every().day.at("21:43").do(main) #毎日何時に起動するか
while True:
  schedule.run_pending()
  time.sleep(60)



   
        
        


