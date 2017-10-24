#-*-coding:utf-8-*-

import course

class CourseFT:
    def __init__(self, enrollment):
        self.course_feature = {}
        # course_enrollment = {key:course_id,[enrollment_ids]}  一个课程所有的enrollment_id
        self.course_enrollment = course.CourseInfo('..\\data\\train\\enrollment_train.csv').course_enrollment_info
        self.enrollment = enrollment
        for course_id in self.course_enrollment:
            enroll_num = len(self.course_enrollment[course_id])
            log_num = 0
            source_sts = [0, 0]
            event_sts = [0 for x in range(0, 7)]
            span = 0  # 这门课程的平均学习天数
            for enrollment_id in self.course_enrollment[course_id]:
                log_num += self.enrollment[enrollment_id][1]
                source_sts = [source_sts[i] + self.enrollment[enrollment_id][i+1] for i in range(0, 2)]
                event_sts = [event_sts[i] + self.enrollment[enrollment_id][i+3] for i in range(0, 7)]
                span += self.enrollment[enrollment_id][10]

            self.course_feature[course_id] = [enroll_num] + \
                                             [int(x/enroll_num) for x in [log_num, span] + source_sts + event_sts]

            # np.savetxt('fuck10.csv', np.array(list(self.enrollment.values()),dtype=npfloat))


