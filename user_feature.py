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
            day_variance=0.0
            log_variance=0.0
            username_courseNum=len(self.user_enrollment[username])
            for enrollment_id in self.user_enrollment[username]:
                span += self.enrollment[enrollment_id][11]
                log_num += self.enrollment[enrollment_id][0]
                source_sts=[source_sts[i]+self.enrollment[enrollment_id][i+1] for i in range(0,2)]
                event_sts=[event_sts[i]+self.enrollment[enrollment_id][i+3] for i in range(0,7)]
                day_num+=self.enrollment[enrollment_id][10]

            span_aver = float(span)  / len(self.user_enrollment[username])
            day_aver=float(day_num) / len(self.user_enrollment[username])
            log_num=float(log_num) / len(self.user_enrollment[username])
            source_sts=[x/len(self.user_enrollment[username]) for x in source_sts]
            event_sts=[x/len(self.user_enrollment[username]) for x in event_sts]

            for enrollment_id in self.user_enrollment[username]:
                day_variance+=(self.enrollment[enrollment_id][10]-day_aver)**2
                log_variance+=(self.enrollment[enrollment_id][0]-log_num)**2
            day_variance/=username_courseNum
            log_variance/=username_courseNum

            self.user_feature[username] = [log_num]+source_sts+event_sts+[span_aver,day_aver,username_courseNum,day_variance,log_variance]

