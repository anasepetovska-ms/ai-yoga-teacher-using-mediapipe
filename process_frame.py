import time
import cv2
import numpy as np
from utils import find_angle, get_landmark_features, draw_text, draw_dotted_line

def isGoodAsana(angles, asana):
    LEFT, RIGHT = "LEFT", "RIGHT"
    if asana == "downdog":
        return 110 < angles[LEFT]["HIP"] < 130 \
            and 130 < angles[LEFT]["KNEE"] < 150 \
            and 140 < angles[LEFT]["ANKLE"] < 160 \
            and 150 < angles[LEFT]["ELBOW"] < 170 \
            and 140 < angles[LEFT]["WRIST"] < 160
    if asana == "tree":
        return (
            (  #right leg up
                0 < angles[LEFT]["HIP"] < 20 \
                and 160 < angles[LEFT]["KNEE"] < 180 \
                and 160 < angles[LEFT]["ANKLE"] < 180 \
            )
            and (
                0 < angles[RIGHT]["HIP"] < 20 \
                and 110 < angles[RIGHT]["KNEE"] < 130 \
                and 85 < angles[RIGHT]["ANKLE"] < 105 \
            )
        ) or (
            (  #left leg up
                0 < angles[RIGHT]["HIP"] < 20 \
                and 160 < angles[RIGHT]["KNEE"] < 180 \
                and 160 < angles[RIGHT]["ANKLE"] < 180 \
            )
            and (
                0 < angles[LEFT]["HIP"] < 20 \
                and 110 < angles[LEFT]["KNEE"] < 130 \
                and 85 < angles[LEFT]["ANKLE"] < 105 \
            )
        )
    else:
        return False

class ProcessFrame:
    def __init__(self, thresholds, flip_frame = False):
        
        # Set if frame should be flipped or not.
        self.flip_frame = flip_frame

        # self.thresholds
        self.thresholds = thresholds

        # Font type.
        self.font = cv2.FONT_HERSHEY_SIMPLEX

        # line type
        self.linetype = cv2.LINE_AA

        # set radius to draw arc
        self.radius = 20

        # Colors in BGR format.
        self.COLORS = {
                        'blue'       : (0, 127, 255),
                        'red'        : (255, 50, 50),
                        'green'      : (0, 255, 127),
                        'light_green': (100, 233, 127),
                        'yellow'     : (255, 255, 0),
                        'magenta'    : (255, 0, 255),
                        'white'      : (255,255,255),
                        'cyan'       : (0, 255, 255),
                        'light_blue' : (102, 204, 255)
                      }



        # Dictionary to maintain the various landmark features.
        self.dict_features = {}
        self.left_features = {
                                'ear'     : 7,
                                'shoulder': 11,
                                'elbow'   : 13,
                                'wrist'   : 15,
                                'hand'    : 19,               
                                'hip'     : 23,
                                'knee'    : 25,
                                'ankle'   : 27,
                                'foot'    : 31
                             }

        self.right_features = {
                                'ear'     : 8,
                                'shoulder': 12,
                                'elbow'   : 14,
                                'wrist'   : 16,
                                'hand'    : 20,
                                'hip'     : 24,
                                'knee'    : 26,
                                'ankle'   : 28,
                                'foot'    : 32
                              }

        self.dict_features['left'] = self.left_features
        self.dict_features['right'] = self.right_features
        self.dict_features['nose'] = 0
    
        self.FEEDBACK_ID_MAP = {
                                0: ('BEND BACKWARDS', 215, (0, 153, 255)),
                                1: ('BEND FORWARD', 215, (0, 153, 255)),
                                2: ('KNEE FALLING OVER TOE', 170, (255, 80, 80)),
                                3: ('SQUAT TOO DEEP', 125, (255, 80, 80))
                               }


    def process(self, frame: np.array, pose):
        play_sound = None
       

        frame_height, frame_width, _ = frame.shape

        # Process the image.
        keypoints = pose.process(frame)

        if keypoints.pose_landmarks:
            ps_lm = keypoints.pose_landmarks

            nose_coord = get_landmark_features(ps_lm.landmark, self.dict_features, 'nose', frame_width, frame_height)
            left_ear_coord, left_shldr_coord, left_elbow_coord, left_wrist_coord, left_hand_coord, left_hip_coord, left_knee_coord, left_ankle_coord, left_foot_coord = \
                                get_landmark_features(ps_lm.landmark, self.dict_features, 'left', frame_width, frame_height)
            right_ear_coord, right_shldr_coord, right_elbow_coord, right_wrist_coord, right_hand_coord, right_hip_coord, right_knee_coord, right_ankle_coord, right_foot_coord = \
                                get_landmark_features(ps_lm.landmark, self.dict_features, 'right', frame_width, frame_height)

            offset_angle = find_angle(left_shldr_coord, right_shldr_coord, nose_coord)

            if offset_angle > self.thresholds['OFFSET_THRESH']:

                cv2.circle(frame, nose_coord, 7, self.COLORS['white'], -1)
                cv2.circle(frame, left_ear_coord, 7, self.COLORS['yellow'], -1)
                cv2.circle(frame, right_ear_coord, 7, self.COLORS['magenta'], -1)
                cv2.circle(frame, left_shldr_coord, 7, self.COLORS['yellow'], -1)
                cv2.circle(frame, right_shldr_coord, 7, self.COLORS['magenta'], -1)

                #----------------------------------------------------------------------------
                
                # Plot landmark points
                cv2.circle(frame, left_shldr_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, left_elbow_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, left_wrist_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, left_hand_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, left_hip_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, left_knee_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, left_ankle_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, left_foot_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)

                cv2.circle(frame, right_shldr_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, right_elbow_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, right_wrist_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, right_hand_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, right_hip_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, right_knee_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, right_ankle_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, right_foot_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)


                #----------------------------------------------------------------------------
                multiplier = -1

                
            #
            # RIGHT
            #

                right_shldr_vertical_angle = find_angle(right_elbow_coord, np.array([right_shldr_coord[0], 0]), right_shldr_coord)
                cv2.ellipse(frame, right_shldr_coord, (30, 30), 
                            angle = 0, startAngle = -90, endAngle = -90+multiplier*right_shldr_vertical_angle, 
                            color = self.COLORS['white'], thickness = 3, lineType = self.linetype)

                draw_dotted_line(frame, right_shldr_coord, start=right_shldr_coord[1]-80, end=right_shldr_coord[1]+20, line_color=self.COLORS['blue'])

                right_elbow_vertical_angle = find_angle(right_wrist_coord, np.array([right_elbow_coord[0], 0]), right_elbow_coord)
                cv2.ellipse(frame, right_elbow_coord, (30, 30), 
                            angle = 0, startAngle = -90, endAngle = -90+multiplier*right_elbow_vertical_angle, 
                            color = self.COLORS['white'], thickness = 3, lineType = self.linetype)

                draw_dotted_line(frame, right_elbow_coord, start=right_elbow_coord[1]-80, end=right_elbow_coord[1]+20, line_color=self.COLORS['blue'])

                right_wrist_vertical_angle = find_angle(right_hand_coord, np.array([right_wrist_coord[0], 0]), right_wrist_coord)
                cv2.ellipse(frame, right_wrist_coord, (30, 30), 
                            angle = 0, startAngle = -90, endAngle = -90+multiplier*right_wrist_vertical_angle, 
                            color = self.COLORS['white'], thickness = 3, lineType = self.linetype)

                draw_dotted_line(frame, right_wrist_coord, start=right_wrist_coord[1]-80, end=right_wrist_coord[1]+20, line_color=self.COLORS['blue'])



                right_hip_vertical_angle = find_angle(right_shldr_coord, np.array([right_hip_coord[0], 0]), right_hip_coord)
                cv2.ellipse(frame, right_hip_coord, (30, 30), 
                            angle = 0, startAngle = -90, endAngle = -90+multiplier*right_hip_vertical_angle, 
                            color = self.COLORS['white'], thickness = 3, lineType = self.linetype)

                draw_dotted_line(frame, right_hip_coord, start=right_hip_coord[1]-80, end=right_hip_coord[1]+20, line_color=self.COLORS['blue'])

                right_knee_vertical_angle = find_angle(right_hip_coord, np.array([right_knee_coord[0], 0]), right_knee_coord)
                cv2.ellipse(frame, right_knee_coord, (20, 20), 
                            angle = 0, startAngle = -90, endAngle = -90-multiplier*right_knee_vertical_angle, 
                            color = self.COLORS['white'], thickness = 3,  lineType = self.linetype)

                draw_dotted_line(frame, right_knee_coord, start=right_knee_coord[1]-50, end=right_knee_coord[1]+20, line_color=self.COLORS['blue'])



                right_ankle_vertical_angle = find_angle(right_knee_coord, np.array([right_ankle_coord[0], 0]), right_ankle_coord)
                cv2.ellipse(frame, right_ankle_coord, (30, 30),
                            angle = 0, startAngle = -90, endAngle = -90 + multiplier*right_ankle_vertical_angle,
                            color = self.COLORS['white'], thickness = 3,  lineType=self.linetype)

                draw_dotted_line(frame, right_ankle_coord, start=right_ankle_coord[1]-50, end=right_ankle_coord[1]+20, line_color=self.COLORS['blue'])


                right_shldr_text_coord_x = frame_width - right_shldr_coord[0] + 10
                right_elbow_text_coord_x = frame_width - right_elbow_coord[0] + 10
                right_wrist_text_coord_x  = frame_width - right_wrist_coord[0] + 10
                right_hip_text_coord_x   = frame_width - right_hip_coord[0] + 10
                right_knee_text_coord_x  = frame_width - right_knee_coord[0] + 15
                right_ankle_text_coord_x = frame_width - right_ankle_coord[0] + 10


            #
            # LEFT
            #

                left_shldr_text_coord_x = left_shldr_coord[0] + 10
                left_elbow_text_coord_x = left_elbow_coord[0] + 10
                left_wrist_text_coord_x  = left_wrist_coord[0] + 10
                left_hip_text_coord_x   = left_hip_coord[0] + 10
                left_knee_text_coord_x  = left_knee_coord[0] + 15
                left_ankle_text_coord_x = left_ankle_coord[0] + 10


                left_shldr_vertical_angle = find_angle(left_elbow_coord, np.array([left_shldr_coord[0], 0]), left_shldr_coord)
                cv2.ellipse(frame, left_shldr_coord, (30, 30), 
                            angle = 0, startAngle = -90, endAngle = -90+multiplier*left_shldr_vertical_angle, 
                            color = self.COLORS['white'], thickness = 3, lineType = self.linetype)

                draw_dotted_line(frame, left_shldr_coord, start=left_shldr_coord[1]-80, end=left_shldr_coord[1]+20, line_color=self.COLORS['blue'])

                left_elbow_vertical_angle = find_angle(left_wrist_coord, np.array([left_elbow_coord[0], 0]), left_elbow_coord)
                cv2.ellipse(frame, left_elbow_coord, (30, 30), 
                            angle = 0, startAngle = -90, endAngle = -90+multiplier*left_elbow_vertical_angle, 
                            color = self.COLORS['white'], thickness = 3, lineType = self.linetype)

                draw_dotted_line(frame, left_elbow_coord, start=left_elbow_coord[1]-80, end=left_elbow_coord[1]+20, line_color=self.COLORS['blue'])

                left_wrist_vertical_angle = find_angle(left_hand_coord, np.array([left_wrist_coord[0], 0]), left_wrist_coord)
                cv2.ellipse(frame, left_wrist_coord, (30, 30), 
                            angle = 0, startAngle = -90, endAngle = -90+multiplier*left_wrist_vertical_angle, 
                            color = self.COLORS['white'], thickness = 3, lineType = self.linetype)

                draw_dotted_line(frame, left_wrist_coord, start=left_wrist_coord[1]-80, end=left_wrist_coord[1]+20, line_color=self.COLORS['blue'])



                left_hip_vertical_angle = find_angle(left_shldr_coord, np.array([left_hip_coord[0], 0]), left_hip_coord)
                cv2.ellipse(frame, left_hip_coord, (30, 30), 
                            angle = 0, startAngle = -90, endAngle = -90+multiplier*left_hip_vertical_angle, 
                            color = self.COLORS['white'], thickness = 3, lineType = self.linetype)

                draw_dotted_line(frame, left_hip_coord, start=left_hip_coord[1]-80, end=left_hip_coord[1]+20, line_color=self.COLORS['blue'])

                left_knee_vertical_angle = find_angle(left_hip_coord, np.array([left_knee_coord[0], 0]), left_knee_coord)
                cv2.ellipse(frame, left_knee_coord, (20, 20), 
                            angle = 0, startAngle = -90, endAngle = -90-multiplier*left_knee_vertical_angle, 
                            color = self.COLORS['white'], thickness = 3,  lineType = self.linetype)

                draw_dotted_line(frame, left_knee_coord, start=left_knee_coord[1]-50, end=left_knee_coord[1]+20, line_color=self.COLORS['blue'])



                left_ankle_vertical_angle = find_angle(left_knee_coord, np.array([left_ankle_coord[0], 0]), left_ankle_coord)
                cv2.ellipse(frame, left_ankle_coord, (30, 30),
                            angle = 0, startAngle = -90, endAngle = -90 + multiplier*left_ankle_vertical_angle,
                            color = self.COLORS['white'], thickness = 3,  lineType=self.linetype)

                draw_dotted_line(frame, left_ankle_coord, start=left_ankle_coord[1]-50, end=left_ankle_coord[1]+20, line_color=self.COLORS['blue'])

                color = self.COLORS['red']
                angles = {
                    "LEFT": {
                        "ELBOW": left_elbow_vertical_angle,
                        "WRIST": left_wrist_vertical_angle,
                        "HIP": left_hip_vertical_angle,
                        "KNEE": left_knee_vertical_angle,
                        "ANKLE": left_ankle_vertical_angle,
                    },
                    "RIGHT": {
                        "ELBOW": left_elbow_vertical_angle,
                        "WRIST": left_wrist_vertical_angle,
                        "HIP": left_hip_vertical_angle,
                        "KNEE": left_knee_vertical_angle,
                        "ANKLE": left_ankle_vertical_angle,
                    }
                }
                if isGoodAsana(angles, "downdog"):
                    color = self.COLORS['light_blue']

                # Join landmarks.
                cv2.line(frame, left_shldr_coord, right_shldr_coord, color, 4, lineType=self.linetype)
                cv2.line(frame, left_hip_coord, right_hip_coord, color, 4, lineType=self.linetype)

                cv2.line(frame, left_shldr_coord, left_elbow_coord, color, 4, lineType=self.linetype)
                cv2.line(frame, left_wrist_coord, left_elbow_coord, color, 4, lineType=self.linetype)
                cv2.line(frame, left_wrist_coord, left_hand_coord, color, 4, lineType=self.linetype)
                cv2.line(frame, left_shldr_coord, left_hip_coord, color, 4, lineType=self.linetype)
                cv2.line(frame, left_knee_coord,  left_hip_coord, color, 4,  lineType=self.linetype)
                cv2.line(frame, left_ankle_coord, left_knee_coord,color, 4,  lineType=self.linetype)
                cv2.line(frame, left_ankle_coord, left_foot_coord, color, 4,  lineType=self.linetype)

                cv2.line(frame, right_shldr_coord, right_elbow_coord, color, 4, lineType=self.linetype)
                cv2.line(frame, right_wrist_coord, right_elbow_coord, color, 4, lineType=self.linetype)
                cv2.line(frame, right_wrist_coord, right_hand_coord, color, 4, lineType=self.linetype)
                cv2.line(frame, right_shldr_coord, right_hip_coord, color, 4, lineType=self.linetype)
                cv2.line(frame, right_knee_coord,  right_hip_coord, color, 4,  lineType=self.linetype)
                cv2.line(frame, right_ankle_coord, right_knee_coord,color, 4,  lineType=self.linetype)
                cv2.line(frame, right_ankle_coord, right_foot_coord, color, 4,  lineType=self.linetype)

                if self.flip_frame:
                    frame = cv2.flip(frame, 1)
                    left_shldr_text_coord_x = frame_width - left_shldr_coord[0] + 10
                    left_elbow_text_coord_x = frame_width - left_elbow_coord[0] + 10
                    left_wrist_text_coord_x  = frame_width - left_wrist_coord[0] + 10
                    left_hip_text_coord_x   = frame_width - left_hip_coord[0] + 10
                    left_knee_text_coord_x  = frame_width - left_knee_coord[0] + 15
                    left_ankle_text_coord_x = frame_width - left_ankle_coord[0] + 10

                cv2.putText(frame, str(int(left_shldr_vertical_angle)), (left_shldr_text_coord_x, left_shldr_coord[1]), self.font, 0.6, self.COLORS['white'], 2, lineType=self.linetype)
                cv2.putText(frame, str(int(left_elbow_vertical_angle)), (left_elbow_text_coord_x, left_elbow_coord[1]), self.font, 0.6, self.COLORS['white'], 2, lineType=self.linetype)
                cv2.putText(frame, str(int(left_wrist_vertical_angle)), (left_wrist_text_coord_x, left_wrist_coord[1]), self.font, 0.6, self.COLORS['white'], 2, lineType=self.linetype)
                cv2.putText(frame, str(int(left_hip_vertical_angle)), (left_hip_text_coord_x, left_hip_coord[1]), self.font, 0.6, self.COLORS['white'], 2, lineType=self.linetype)
                cv2.putText(frame, str(int(left_knee_vertical_angle)), (left_knee_text_coord_x, left_knee_coord[1]), self.font, 0.6, self.COLORS['white'], 2, lineType=self.linetype)
                cv2.putText(frame, str(int(left_ankle_vertical_angle)), (left_ankle_text_coord_x, left_ankle_coord[1]), self.font, 0.6, self.COLORS['white'], 2, lineType=self.linetype)

                cv2.putText(frame, str(int(right_shldr_vertical_angle)), (right_shldr_text_coord_x, right_shldr_coord[1]), self.font, 0.6, self.COLORS['white'], 2, lineType=self.linetype)
                cv2.putText(frame, str(int(right_elbow_vertical_angle)), (right_elbow_text_coord_x, right_elbow_coord[1]), self.font, 0.6, self.COLORS['white'], 2, lineType=self.linetype)
                cv2.putText(frame, str(int(right_wrist_vertical_angle)), (right_wrist_text_coord_x, right_wrist_coord[1]), self.font, 0.6, self.COLORS['white'], 2, lineType=self.linetype)
                cv2.putText(frame, str(int(right_hip_vertical_angle)), (right_hip_text_coord_x, right_hip_coord[1]), self.font, 0.6, self.COLORS['white'], 2, lineType=self.linetype)
                cv2.putText(frame, str(int(right_knee_vertical_angle)), (right_knee_text_coord_x, right_knee_coord[1]), self.font, 0.6, self.COLORS['white'], 2, lineType=self.linetype)
                cv2.putText(frame, str(int(right_ankle_vertical_angle)), (right_ankle_text_coord_x, right_ankle_coord[1]), self.font, 0.6, self.COLORS['white'], 2, lineType=self.linetype)

            else:
                dist_l_sh_hip = abs(left_foot_coord[1] - left_shldr_coord[1])
                dist_r_sh_hip = abs(right_foot_coord[1] - right_shldr_coord[1])

                shldr_coord = None
                elbow_coord = None
                wrist_coord = None
                hip_coord = None
                knee_coord = None
                ankle_coord = None
                foot_coord = None

                if dist_l_sh_hip > dist_r_sh_hip:
                    shldr_coord = left_shldr_coord
                    elbow_coord = left_elbow_coord
                    wrist_coord = left_wrist_coord
                    hand_coord  = left_hand_coord
                    hip_coord   = left_hip_coord
                    knee_coord  = left_knee_coord
                    ankle_coord = left_ankle_coord
                    foot_coord  = left_foot_coord

                    multiplier = -1
                
                else:
                    shldr_coord = right_shldr_coord
                    elbow_coord = right_elbow_coord
                    wrist_coord = right_wrist_coord
                    hand_coord  = right_hand_coord
                    hip_coord   = right_hip_coord
                    knee_coord  = right_knee_coord
                    ankle_coord = right_ankle_coord
                    foot_coord  = right_foot_coord

                    multiplier = 1
                    

                # ------------------- Vertical Angle calculation --------------
                
                shldr_vertical_angle = find_angle(elbow_coord, np.array([shldr_coord[0], 0]), shldr_coord)
                cv2.ellipse(frame, shldr_coord, (30, 30), 
                            angle = 0, startAngle = -90, endAngle = -90+multiplier*shldr_vertical_angle, 
                            color = self.COLORS['white'], thickness = 3, lineType = self.linetype)

                draw_dotted_line(frame, shldr_coord, start=shldr_coord[1]-80, end=shldr_coord[1]+20, line_color=self.COLORS['blue'])

                hip_vertical_angle = find_angle(shldr_coord, np.array([hip_coord[0], 0]), hip_coord)
                cv2.ellipse(frame, hip_coord, (30, 30), 
                            angle = 0, startAngle = -90, endAngle = -90+multiplier*hip_vertical_angle, 
                            color = self.COLORS['white'], thickness = 3, lineType = self.linetype)

                draw_dotted_line(frame, hip_coord, start=hip_coord[1]-80, end=hip_coord[1]+20, line_color=self.COLORS['blue'])




                knee_vertical_angle = find_angle(hip_coord, np.array([knee_coord[0], 0]), knee_coord)
                cv2.ellipse(frame, knee_coord, (20, 20), 
                            angle = 0, startAngle = -90, endAngle = -90-multiplier*knee_vertical_angle, 
                            color = self.COLORS['white'], thickness = 3,  lineType = self.linetype)

                draw_dotted_line(frame, knee_coord, start=knee_coord[1]-50, end=knee_coord[1]+20, line_color=self.COLORS['blue'])



                ankle_vertical_angle = find_angle(knee_coord, np.array([ankle_coord[0], 0]), ankle_coord)
                cv2.ellipse(frame, ankle_coord, (30, 30),
                            angle = 0, startAngle = -90, endAngle = -90 + multiplier*ankle_vertical_angle,
                            color = self.COLORS['white'], thickness = 3,  lineType=self.linetype)

                draw_dotted_line(frame, ankle_coord, start=ankle_coord[1]-50, end=ankle_coord[1]+20, line_color=self.COLORS['blue'])

                # ------------------------------------------------------------
        
                
                # Join landmarks.
                cv2.line(frame, shldr_coord, elbow_coord, self.COLORS['light_blue'], 4, lineType=self.linetype)
                cv2.line(frame, wrist_coord, elbow_coord, self.COLORS['light_blue'], 4, lineType=self.linetype)
                cv2.line(frame, wrist_coord, hand_coord, self.COLORS['light_blue'], 4, lineType=self.linetype)
                cv2.line(frame, shldr_coord, hip_coord, self.COLORS['light_blue'], 4, lineType=self.linetype)
                cv2.line(frame, knee_coord, hip_coord, self.COLORS['light_blue'], 4,  lineType=self.linetype)
                cv2.line(frame, ankle_coord, knee_coord,self.COLORS['light_blue'], 4,  lineType=self.linetype)
                cv2.line(frame, ankle_coord, foot_coord, self.COLORS['light_blue'], 4,  lineType=self.linetype)
                
                # Plot landmark points
                cv2.circle(frame, shldr_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, elbow_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, wrist_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, hand_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, hip_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, knee_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, ankle_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, foot_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)

                hip_text_coord_x = hip_coord[0] + 10
                knee_text_coord_x = knee_coord[0] + 15
                ankle_text_coord_x = ankle_coord[0] + 10

                if self.flip_frame:
                    frame = cv2.flip(frame, 1)
                    hip_text_coord_x = frame_width - hip_coord[0] + 10
                    knee_text_coord_x = frame_width - knee_coord[0] + 15
                    ankle_text_coord_x = frame_width - ankle_coord[0] + 10


                cv2.putText(frame, str(int(shldr_vertical_angle)), (hip_text_coord_x, hip_coord[1]), self.font, 0.6, self.COLORS['white'], 2, lineType=self.linetype)
                cv2.putText(frame, str(int(hip_vertical_angle)), (hip_text_coord_x, hip_coord[1]), self.font, 0.6, self.COLORS['white'], 2, lineType=self.linetype)
                cv2.putText(frame, str(int(knee_vertical_angle)), (knee_text_coord_x, knee_coord[1]+10), self.font, 0.6, self.COLORS['white'], 2, lineType=self.linetype)
                cv2.putText(frame, str(int(ankle_vertical_angle)), (ankle_text_coord_x, ankle_coord[1]), self.font, 0.6, self.COLORS['white'], 2, lineType=self.linetype)

        else:

            if self.flip_frame:
                frame = cv2.flip(frame, 1)

        return frame, play_sound