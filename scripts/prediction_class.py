from operator import inv
import numpy as np

class Domain:
    def __init__(self, delim, score = 0):
        if ";" in delim:
            domains = delim.split(";")
            delim0 = domains[0].split("-")
            if delim0[0]  == "":
                delim0 = delim0[1:]
            delim1 = domains[1].split("-")
            self.delim = delim0 + delim1
            self.name = str(delim0[0]) + "-"+ str(delim0[1]) + ";" + str(delim1[0]) + "-"+ str(delim1[1])
            arange1 = np.arange(int(delim0[0]), int(delim0[1]) + 1)
            arange2 = np.arange(int(delim1[0]), int(delim1[1]) + 1)
            zeros = np.full( int(delim1[0])   - int(delim0[1]), -1)

            self.arange = np.concatenate((arange1,zeros, arange2))
        else:
            try:
                self.delim = delim.split("-")
            except:
                self.delim = delim
            
            if self.delim[0]  == "":
                self.delim = self.delim[1:]
            if self.delim[0] == "1H":
                self.delim[0] = '1'
            self.name = str(self.delim[0]) + "-"+ str(self.delim[1])
            self.arange = np.arange(int(self.delim[0]), int(self.delim[1]) + 1)
        # print(delim)

        # self.delim = delim.split('-')
        
        # self.name = str(self.delim[0]) + "-"+ str(self.delim[1])
        # self.arange = np.arange(int(self.delim[0]), int(self.delim[1]) + 1)
        self.score = score
        self.cov_gen = 0
        self.bench = []
        self.cov_bench = []

    def set_cov_gen(self, cov_gen):
        self.cov_gen = cov_gen
        
    def set_bench(self, bench):
        self.bench.append(bench)
        
    def set_cov_bench(self, cov_bench):
        self.cov_bench.append(cov_bench)

    def get_delim(self):
        return self.delim

    def get_score(self):
        return self.score
        
    def get_cov_gen(self):
        return self.cov_gen
        
    def get_bench(self):
        return self.bench
        
    def get_cov_bench(self):
        return self.cov_bench

    def get_arange(self):
        return self.arange

    def get_info_bench(self):
        if len(self.cov_bench) != 0:
            tab = 3 - len(self.cov_bench)
        else:
            tab = 4 
        return ("{} \t{:5s}\t{:15s}\t\t{}{}{}\n".format(self.delim, str(round(self.score,2)), str(self.cov_gen), self.bench, "\t"*tab,self.cov_bench))

    def get_info(self):
        return ("{} \t{:5s}\t{:15s}\n".format(self.delim, str(self.score), str(self.cov_gen)))

    # def get_coverage(self, obj_domain, size):
    #     return round(np.in1d(obj_domain.get_arange(),
    #      self.get_arange()).sum() / size * 100,2)



    def get_coverage(self, obj_domain):
        d0 = abs(int(self.delim[0]) - int(obj_domain.delim[0]))
        d1 = abs(int(self.delim[1]) - int(obj_domain.delim[1]))
        miss =  d0 + d1
        max_right = max(int(self.delim[1]), int(obj_domain.delim[1]))
        min_left= min(int(self.delim[0]), int(obj_domain.delim[0]))
        length = max_right - min_left
        if length == 0:
            return 100
        return (1 - miss / length) * 100

    # def get_coverage(self, obj_domain, min, size):
    #     ref0 = int(self.delim[0])
    #     ref1 = int(self.delim[1])

    #     pred0 = int(obj_domain.delim[0])
    #     pred1 = int(obj_domain.delim[1])

    #     first_half_ref = np.full(abs(ref0 - min ), -1)
    #     first_half_pred = np.full(abs(pred0 - min ), -1)

    #     second_half_ref = np.full(abs(ref1 - size ), -1)
    #     second_half_pred = np.full(abs(pred1 - size ), -1)
    #     print(self, obj_domain)
    #     print(first_half_ref.shape, second_half_ref.shape,first_half_pred.shape, second_half_pred.shape)
    #     ref = np.concatenate((first_half_ref, self.get_arange(),second_half_ref ))
    #     pred = np.concatenate((first_half_pred, self.get_arange(),second_half_pred ))
    #     print(np.in1d(ref, pred, invert= True).sum())

    #     print((1 - (np.in1d(ref, pred, invert= True).sum()  / size)) * 100)
    #     return  (1 - (np.in1d(ref, pred, invert= True).sum()  / size)) * 100

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name
    
    def __lt__(self, other):
        return self.delim < other.delim

    def __gt__(self, point):
        return self.delim > point.delim

def pred_to_object(results, first_pos):

    predictions = []
    for i in range(len(results["rois"])):
        x1, y1, x2, y2 = results["rois"][i]
        res1 = min(x1,y1)
        res2 = max(x2,y2)
        try:
            first_pos = int(first_pos)
        except:
            first_pos = int(first_pos[:-1])
        res1 = int(res1) +  int(first_pos) - 1
        res2 = int(res2) +  int(first_pos) - 1
        predictions.append(Domain(delim = (res1, res2),
         score = results["scores"][i]))
    return predictions


if __name__ == "__main__":
    delim = "254-499"
    ref = Domain(delim = delim )
    print("Ref : ",ref)

    delim = "234-489"
    pred = Domain(delim = delim )
    print("Method : ",pred)

    size  = 499
    print( '1 - ',ref.get_uncov(pred) ,'/', size, "=", (1  - ref.get_uncov(pred) / size) )
    # print(ref.get_coverage(pred))