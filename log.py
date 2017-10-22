import csv


class LogInfo:
    def __init__(self, filename):
        self.enrollment_info = {}

        with open(filename, 'r') as f:
            info = list(csv.reader(f))[1:]

            for row in info:
                enrollment_id, time, source, event = row[0:4]
                if(enrollment_id not in self.enrollment_info):
                    self.enrollment_info[enrollment_id] = [[time, source, event]]
                else:
                    self.enrollment_info[enrollment_id].append([time, source, event])

print(1)