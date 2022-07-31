


import sys


flist = open("all_train_images.txt").readlines()
rlist = open("all_train_results.txt").readlines()


pred_num = [0] * 5
pred_dict = {0:[], 1:[], 2:[], 3:[], 4:[],}
print(pred_dict)

thres= 0.95
if len(sys.argv) >= 2:
    thres = float(sys.argv[1])

with open("updated_train_images.txt", "w+") as f:
    # f.write("number,label\n")
    # /diskssd0/datasets/yunwen/test/10000.png
    flist = [itf.strip() for itf in flist]
    print(len(flist))
    print(len(rlist))
    out_lines = []
    for itf,itr in zip(flist, rlist):
        pred, score = itr.strip().split(",")
        pred = int(pred)
        score = float(score)
        if score < thres:
            continue
        pred_num[pred] += 1
        pred_dict[pred].append([itf, int(pred)])
        # out_lines.append([itf, int(pred)])
    print(pred_num)
    mlen = min(pred_num)
    out_lines = pred_dict[0][:mlen] + \
                pred_dict[1][:mlen] + \
                pred_dict[2][:mlen] + \
                pred_dict[3][:mlen] + \
                pred_dict[4][:mlen]
    print(len(out_lines))

    # out_lines = sorted(out_lines, key=lambda x:x[0])
    out_lines = ["{},{}\n".format(it[0],it[1]) for it in out_lines]


    f.writelines(out_lines)




