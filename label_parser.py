# Import the os module, for the os.walk function
import os
 
# Set the directory you want to start from
rootDir = './Emotion'
parallelDir = './cohn-kanade-images'
output = open('emotion_labels.txt', 'w');
output_pic_names = open('associated_pic_filenames.txt', 'w');
for dirName, subdirList, fileList in os.walk(rootDir):
    for fname in fileList:
				print dirName + "/" + fname
				label_reader = open(dirName + "/" + fname, 'r')
				raw_data = label_reader.readline()[3:-13]
				output.write(raw_data + "\n")
				pic = parallelDir + "/" + dirName[10:] + "/" + fname[:-12] + ".png\n"
				output_pic_names.write(pic)
