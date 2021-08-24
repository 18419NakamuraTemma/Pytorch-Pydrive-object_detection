import os
import time
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import detect
import glob
import shutil

gauth = GoogleAuth()
gauth.LocalWebserverAuth()


drive = GoogleDrive(gauth)

drive_tempfolder_id = drive.ListFile({'q': 'title = "tmp"'}).GetList()[0]['id'] # 画像の場所
