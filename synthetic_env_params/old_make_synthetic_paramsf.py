import math

# GEN FEATS

# large: alpha=1
# small: alpha=0.5

outflist = [ "delayed_effects_large", "delayed_effects"]
kappalist = [1, 0.5]
effectlist = [0, 0.2, 0.4]

past_action_len = 8 # past 8 actions

#Parameters are for covariates in the following order:
    # [8_prev_action, 7_prev_action, ..., 1_prev_action, \
        # 8_prev_action*1_prev_reward, 1_prev_action*1_prev_reward, ..., 1_prev_action*1_prev_reward \
        # action, action*1_prev_reward]

for outf, kappa in zip(outflist, kappalist):
    for effect in effectlist:
        outf_str = outf
        if effect != 0:
            outf_str += "_te={}".format(effect).replace(".", "")
        print(outf_str)
        with open(outf_str+".txt", 'w') as f:
            decaying = [ kappa*round( math.exp(-s), 3 ) for s in range(past_action_len) ]
            decaying.reverse()
            all_params = decaying + [x*0.5 for x in decaying] + [effect, effect*0.5]
            #base = decaying + [effect]
            #tmp = [x*0.5 for x in base]
            #all_params = base + tmp
            f.write( ",".join( [str(x) for x in all_params ]) + "\n" )
