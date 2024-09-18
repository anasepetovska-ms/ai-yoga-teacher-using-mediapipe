import json
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import random

popular = [
    'Ardha Chandrasana',
    'Baddha Konasana',
    'Adho Mukha Svanasana',
    'Natarajasana',
    'Trikonasana',
    'UtkataKonasana',
    'Veerabhadrasana',
    'Vrukshasana',
    ]

def load_poses_dataset():
    # reading the JSON data using json.load()
    file = 'yogaposes.json'
    with open(file) as train_file:
        dict_train = json.load(train_file)

    #converting json dataset from dictionary to dataframe

    data = pd.DataFrame.from_dict(dict_train)

    poses_columns = ['id','sanskrit_name','english_name','procedure', 'target_body_parts', 'contraindications', 'benefits']

    poses = data.filter(items=poses_columns)
    poses = data.reindex(columns=poses_columns)
    poses = data.get(poses_columns).copy()
    return poses

def define_pose_type(poses):
    positionVerbs = ['Stand', 'Standing', 'Lie', 'Sit', 'sitting', 'Base']

    types= '|'.join(positionVerbs)
    poses['procedure_str'] = poses['procedure'].map(str)

    poses['type']=poses.procedure_str.str.findall(types,  flags=re.IGNORECASE)


def create_pose_type_list(poses, type):
    posesList = []
    for i in range(0, len(poses)):
        if (isinstance(poses.iloc[i]['type'], list) and (type in poses.iloc[i]['type'])): # or 'Base' in poses.iloc[i]['type'])):
            
            cues = poses.iloc[i]['procedure_str'].replace('[','').replace(']','').replace("'", '').replace(',','')
            posesList.append(poses.iloc[i]['english_name'] + '. ' + cues)
    return posesList

def generate_sequence_index(flowSequence, action, max, poses_count):
        for i in range(0, max):
            index = random.randint(0, poses_count-1)
            flowSequence.append((action, index))

def generate_sequence():

    poses = load_poses_dataset()
    define_pose_type(poses)
   
    sitting_poses = create_pose_type_list(poses, 'Sit')
    standing_poses = create_pose_type_list(poses, 'Stand')
    laying_poses = create_pose_type_list(poses, 'Lie')

    flow_sequence = []
    generate_sequence_index(flow_sequence, 'Sit', 0, len(sitting_poses) )
    generate_sequence_index(flow_sequence, 'Stand', 2, len(standing_poses) )
    generate_sequence_index(flow_sequence, 'Sit', 0, len(laying_poses) )

    # return flow_sequence

    flow = []

    for i in range(0, len(flow_sequence)):
        if (i == 0):
            flow.append("Welcome to your yoga flow. Begin with ")
        if (i > 0):
            flow.append("Next is ")
        if flow_sequence[i][0] == 'Sit':
            flow.append(sitting_poses[flow_sequence[i][1]])
        if flow_sequence[i][0] == 'Stand':
            flow.append(standing_poses[flow_sequence[i][1]])
        if flow_sequence[i][0] == 'Lie':
            flow.append(laying_poses[flow_sequence[i][1]])

    return flow



def generate_sequence_text(): 
    sequence = generate_sequence()
    # text = ','.join(sequence)
    return sequence
