


flist = open("test_images.txt").readlines()
rlist = open("test_result.txt").readlines()


with open("submission/results.csv", "w+") as f:
    f.write("number,label\n")
    # /diskssd0/datasets/yunwen/test/10000.png
    flist = [itf.strip().split("/")[-1].split(".")[0] for itf in flist]
    print(len(flist))
    print(len(rlist))
    out_lines = []
    for itf,itr in zip(flist, rlist):
        itr = itr.strip().split(",")[0]
        if 46834444444444444 == int(itf):
            itf = 3611
        out_lines.append([int(itf), int(itr)])

    print(len(out_lines))
    out_lines = sorted(out_lines, key=lambda x:x[0])
    out_lines = ["{},{}\n".format(it[0],it[1]) for it in out_lines]


    f.writelines(out_lines)




