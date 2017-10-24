#-*-coding:utf-8-*-

import enrollment_feature
import course_feature
import numpy as np


class merge:
    def __init__(self):
        self.features = []
        enrollment_Feature = enrollment_feature.EnrollmentFT().user_course
        courseFT = course_feature.CourseFT(enrollment_Feature)
        course_Feature = courseFT.course_feature
        enrollment_course = courseFT.course_enrollment
        # for enrollment_id in enrollment_Feature.keys():
        #     for course_id in enrollment_course.keys():
        #         if(enrollment_id in enrollment_course[course_id]):
        #             self.features.append(enrollment_Feature[enrollment_id] + course_Feature[course_id])
        for course_id in enrollment_course:
            for enroll_id in enrollment_course[course_id]:
                self.features.append(enrollment_Feature[enroll_id] + course_Feature[course_id])

        # np.savetxt('features.csv', np.array(list(self.features), fmt='%.4f'))
    # np.savetxt('features.csv', fmt='%.4f', dtype=float)

f = merge().features
for i in f:
    print(i)