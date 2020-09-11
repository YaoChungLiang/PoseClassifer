import math
import json
import matplotlib.pyplot as plt
from os import path
import sys
import time
import cfg

Cfg = cfg.Config()

def getMarkNumMap(labelFile):
	res = []
	mark_set = set()
	mark_dict = dict()
	rev_mark_dict = dict()
	with open(labelFile) as json_file:
		data = json.load(json_file)
	for i in data:
		#print(i["StartFrame"])
		s = i["StartFrame"]
		e = i["EndFrame"]
		mark = i["MotorActionFriendlyName"]
		res.append([mark,s,e])
		mark_set.add(mark)
	classes = 1
	for j in mark_set:
		mark_dict[j] = classes
		rev_mark_dict[classes] = j
		classes += 1
	return (res,mark_dict,rev_mark_dict)

def getTargetFrame(dataFile):
  with open(Cfg.dataPath) as json_file:
    data = json.load(json_file)
  img_num = 34029
  PreNum = 1
  counter = 0
  fullList = []
  for i in data['AnnotationList']:
    CurNum = i['FrameNumber']
    if CurNum > img_num:
      break
    if CurNum == PreNum:
      counter+=1
    else:
        #print(counter)
        if counter >= Cfg.ptsNum:
          fullList.append(PreNum)
        PreNum = CurNum
        counter = 0
  return fullList

def extractDataLabel(infile):
  m = math.sqrt(3)
  # poses = [sit :0, lie:1 , stand:2, kneel down: 3 , crawl: 4]
  line = [[0,1],[1,3],[3,5],[0,2],[2,4],[4,6],[1,2],[2,8],[1,7],[7,8],[7,9],[9,11],[8,10],[10,12]]
  # colors = [01,13,35,02,24,46,   12,28,17,78,  79,911,810,1012]
  colors = ['b','r','b','r','b','r', 'k','b','r','k', 'b','r','r','b']	
  PtsNum = 13
  
  target_list = getTargetFrame(Cfg.dataPath)
  #print(len(target_list))
  #sys.exit()
  # generate marker
  
  marker, mark_dict, rev_mark_dict = getMarkNumMap(Cfg.labelPath)
  mark_dict["None"] = 0
  rev_mark_dict[0] = "None"
  #print(marker)
  #return None
  
  target_img_num = len(target_list)
  target_label_list = [[0]]*target_img_num
  img_num = 34029
  label_list = [[0]]*(img_num+1)
  #print(label_list)
  #sys.exit()
	
  for i in marker:
	  if i[1]>img_num:
		  break
	  for j in range(i[1],i[2]+1):
		  label_list[j] = [ mark_dict[ i[0] ] ]
  j = 0
  for i in target_list:
	  target_label_list[j] = label_list[i]
	  j += 1
  label_list.pop(0)

  target_data_loader = [[]]*target_img_num
  data_loader = [[]]*(img_num+1)



  with open(Cfg.dataPath) as json_file:
    		data = json.load(json_file)
 
  PreNumber = 1
  Pre2Dpts = []

  Pre3Dpts = []
  for i in range(PtsNum):
	  Pre2Dpts.append((-1,-1))
	  Pre3Dpts.append((-1,-1,-1))

  counter = 0

  for i in data['AnnotationList']:
	  x_2D = []
	  y_2D = []

	  x_3D = []
	  y_3D = []
	  z_3D = []

	  TempNumber = i['FrameNumber']
	  if TempNumber>(img_num+1):
		  break
	  if TempNumber == PreNumber:
		
		  Pre2Dpts[i['Part']]=(i['LocationX'],i['LocationY'])
		  Pre3Dpts[i['Part']]=(i['WorldX'],i['WorldY'],i['WorldZ'])
		
	  else:
		  data_loader[PreNumber] = []
		  for ln in line:
			if not isinstance(Pre3Dpts[ln[1]][0],basestring) and not isinstance(Pre3Dpts[ln[0]][0],basestring) and sum(Pre3Dpts[ln[1]])!=-3 and sum(Pre3Dpts[ln[0]])!=-3:
				xVec = Pre3Dpts[ln[1]][0] - Pre3Dpts[ln[0]][0]
				yVec = Pre3Dpts[ln[1]][1] - Pre3Dpts[ln[0]][1]
				zVec = Pre3Dpts[ln[1]][2] - Pre3Dpts[ln[0]][2]
				t = math.sqrt(xVec*xVec + yVec*yVec + zVec*zVec)
				if t!=0:
					data_loader[PreNumber].extend([xVec/t,yVec/t,zVec/t,1])
				else:
					data_loader[PreNumber].extend([-1/m,-1/m,-1/m,0])
			
					
			else:
				
				data_loader[PreNumber].extend([-1/m,-1/m,-1/m,0])


		  for p in Pre2Dpts:
			  if p[0] != -1 and p[1] != -1:
				  x_2D.append(p[0])
				  y_2D.append(p[1])
			  #x.append(p[0]) if p[0] > 0 else pass
			  #y.append(p[1]) if p[1] > 0 else pass
		  


		
		  # output normalized vectors
				

		  # init
		  x_2D = []
		  y_2D = []
		  x_3D = []
		  y_3D = []
		  z_3D = []
		  Pre2Dpts = []
		  Pre3Dpts = []
		  for _ in range(PtsNum):
			  Pre2Dpts.append((-1,-1))
			  Pre3Dpts.append((-1,-1,-1))
		  Pre2Dpts[i['Part']] = (i['LocationX'],i['LocationY'])
		  Pre3Dpts[i['Part']] = (i['WorldX'],i['WorldY'],i['WorldZ'])
		  PreNumber = TempNumber
		  #counter += 1
  j = 0
  for i in target_list:
  	target_data_loader[j] = data_loader[i]
  	j += 1

  data_loader.pop(0)	
  # post-process training data:
  for i in range(len(target_data_loader)):
  	if target_data_loader[i]==[]:
		  target_data_loader[i]=[-1/m,-1/m,-1/m,0]*14
		
  for i in range(len(data_loader)):
  	if data_loader[i]==[]:
		  data_loader[i]=[-1/m,-1/m,-1/m,0]*14
  
  return (data_loader,label_list)

if __name__ == "__main__":
	infile = 'MIMM5021-IR-local-annotation-2d-3d-gray.json'
	train_label(infile)
