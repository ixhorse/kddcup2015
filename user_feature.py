#-*-coding:utf-8-*-

import user

class UserFT:
    def __init__(self,enrollment):
        self.user_feature = {}
        self.user_enrollment = user.UserInfo('../data/train/enrollment_train.csv').user_enrollment_info
        self.enrollment=enrollment
        for username in self.user_enrollment.keys():
            log_num = 0
            source_sts = [0, 0]
            event_sts = [0 for x in range(0, 7)]
            span = 0
            day_num=0
            len_user_enrollment=len(self.user_enrollment[username])
            for enrollment_id in self.user_enrollment[username]:
                span += self.enrollment[enrollment_id][11]
                log_num += self.enrollment[enrollment_id][1]
                source_sts=[source_sts[i]+self.enrollment[enrollment_id][i+1] for i in range(0,2)]
                event_sts=[event_sts[i]+self.enrollment[enrollment_id][i+3] for i in range(0,7)]
                day_num+=self.enrollment[enrollment_id][10]
            span_aver = float(span)  / len_user_enrollment
            day_aver=float(day_num) / len_user_enrollment
            log_num=float(log_num) / len_user_enrollment
            source_sts=[x/len_user_enrollment for x in source_sts]
            event_sts=[x/len_user_enrollment for x in event_sts]
            

            self.user_feature[username] = [log_num]+source_sts+event_sts+[span_aver,day_aver]

