def check_range(model,inp,out):
    if model.stride[0]==2 or model.quan_w_fn.bit==8:
        return
    # print("begin")
    # print(model)
    x_q= model.quan_a_fn(inp[0])
    x_q_alpha= model.quan_a_fn.alpha
    w_q=model.quan_w_fn(model.weight)
    w_q_alpha=model.quan_w_fn.alpha
    x_q=x_q/x_q_alpha
    w_q=w_q*w_q_alpha
    # print(inp[0].shape)
    # print(model.weight.shape)
    # print("x_q_alpha: ",x_q_alpha)
    # print("w_q_alpha: ",w_q_alpha)
    # N,C,W,H
    # K,C,R,R
    # print("x_q_shape",x_q.shape)
    # print("w_q_shape",w_q.shape)
    # print("x_q: ",torch.unique(x_q))
    # print("w_q: ",torch.unique(w_q[0]))
    padded_x_q=F.pad(x_q,(1,1,1,1))
    # print("x_q_shape",x_q.shape)
    max_max_num,max_min_num,max_mean_num=0,0,0
    max_bit,min_bit,mean_bit=0,0,0
    for n in range(x_q.shape[0]):
        for k in range(w_q.shape[0]):
            # print("-------------k:",k)
            for w in range(x_q.shape[2]):
                for h in range(x_q.shape[3]):
                    part_sum=torch.mul(padded_x_q[n,:,w:w+3,h:h+3],w_q[k,:,:,:])
                    tmp_max_mean_num,current=0,0
                    for i in part_sum.flatten():
                        current=current+i
                        tmp_max_mean_num=max(tmp_max_mean_num,abs(current))
                    tmp_mean_bit=torch.log2(tmp_max_mean_num).ceil()
                    mean_bit=max(mean_bit,tmp_mean_bit)
                    max_mean_num=max(max_mean_num,tmp_max_mean_num)
                    negative_part_sum=part_sum[part_sum<0]
                    positive_part_sum=part_sum[part_sum>0]
                    tmp_max_max_num=torch.sum(torch.abs(negative_part_sum))
                    tmp_max_max_num=max(tmp_max_max_num,torch.sum(positive_part_sum))
                    tmp_max_bit=torch.log2(tmp_max_max_num).ceil()
                    max_bit=max(max_bit,tmp_max_bit)
                    max_max_num=max(max_max_num,tmp_max_max_num)
                    tmp_max_min_num=torch.max(torch.abs(part_sum))
                    tmp_max_min_num=max(tmp_max_min_num,part_sum.sum())
                    tmp_min_bit=torch.log2(tmp_max_min_num).ceil()
                    min_bit=max(min_bit,tmp_min_bit)
                    max_min_num=max(max_min_num,tmp_max_min_num)
                    # print("part_sum: ",part_sum.shape)
    print("In layer: ",model)
    print("max num: ",max_max_num)
    print("min num: ",max_min_num)
    print("mean num: ",max_mean_num)
    print("max_bit: ",max_bit)
    print("min_bit: ",min_bit)
    print("mean_bit: ",mean_bit)
    # print("input: ",inp[0].min(),inp[0].max())
    # print("output: ",out[0].min(),out[0].max())
    # print(out.flatten()[0:20])

def hook(model,fn):
    for name, module in model.named_modules():
        if isinstance(module, QConv2d):
            module.register_forward_hook(fn)