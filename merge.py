#-*-coding:utf-8-*-

import enrollment_feature
import course_feature
import numpy as np
import user_feature
import user


class merge:
    def __init__(self):
        self.features=[]
        self.enrollment_Feature=enrollment_feature.EnrollmentFT().user_course
        courseFT = course_feature.CourseFT(self.enrollment_Feature)
        course_Feature = courseFT.course_feature
        UserFT = user_feature.UserFT(self.enrollment_Feature)
        user_Feature = UserFT.user_feature
        Info=user.UserInfo('../data/train/enrollment_train.csv').enrollment_train
        for row in Info:
            enrollment_id,username,course_id=row
            self.features.append(self.enrollment_Feature[enrollment_id]+course_Feature[course_id]+user_Feature[username])