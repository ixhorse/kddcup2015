#-*-coding:utf-8-*-
import course
import numpy as np
import enrollment_feature
import os

class CourseFT:
    def __init__(self, enrollment, log_time):
        self.course_feature = {}
        # course_enrollment = {key:course_id,[enrollment_ids],begin,to}  一个课程所有的enrollment_id
        course_info = course.CourseInfo('../data/train/enrollment_train.csv', '../data/date.csv')
        course_enrollment = course_info.course_enrollment_info
        course_date = course_info.course_date
        self.enrollment = enrollment
        for course_id in course_enrollment:
            enroll_num = len(course_enrollment[course_id])
            log_num = 0
            source_sts = [0, 0]
            event_sts = [0 for x in range(0, 7)]
            day_num = 0
            span = 0
            log_list = []
            day_list = []
            for enrollment_id in course_enrollment[course_id]:
                log_num += enrollment[enrollment_id][1]
                log_list.append(enrollment[enrollment_id][1])
                source_sts = [source_sts[i] + enrollment[enrollment_id][i+1] for i in range(0, 2)]
                event_sts = [event_sts[i] + enrollment[enrollment_id][i+3] for i in range(0, 7)]
                day_num += enrollment[enrollment_id][10]
                day_list.append(enrollment[enrollment_id][10])
                span += enrollment[enrollment_id][11]

                time_list = log_time[enrollment_id]
                t_end = time_list[-1]
                t_to = course_date[course_id][1]
                interval_end_to = (t_to - t_end).days
                log_num_last_week = 0
                for day in range(0, len(time_list))[::-1]:
                    if (t_to - time_list[day]).days > 8:
                        break
                    if 0 <= (t_to - time_list[day]).days < 7:
                        log_num_last_week += 1

                self.enrollment[enrollment_id].append(interval_end_to)
                self.enrollment[enrollment_id].append(log_num_last_week)
            self.course_feature[course_id] = [enroll_num] + \
                                             [int(x/enroll_num) for x in [log_num, day_num, span] + [np.var(log_list), np.var(day_list)] + source_sts + event_sts]
            




