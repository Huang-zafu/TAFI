"""Generate controlled synthetic graphs for structural-protection evaluation."""
import numpy as np
GRAPH_KEYS=('x','edge_index','y','train_mask','val_mask','test_mask')

def generate_relational(n_targets,environments,*,split,seed,stream,p_invariant=.95,
                        sigma_source=.25,sigma_target=.5,rel_amplitude=.5,rel_sigma=.75):
    if split not in ['train','development']:raise ValueError('Invalid split')
    if n_targets<=0 or not environments or n_targets%(4*len(environments)):raise ValueError('Target count must be divisible by four times the environment count')
    if not 0<=p_invariant<=1 or any(not 0<=q<=1 for _,q in environments):raise ValueError('Invalid probability')
    if sigma_source<=0 or sigma_target<0 or rel_sigma<=0 or rel_amplitude<=0:raise ValueError('Invalid noise/code parameter')
    rng=np.random.default_rng(np.random.SeedSequence(seed,spawn_key=(stream,)))
    per=n_targets//len(environments)
    y_template=np.tile(np.array([-1,-1,1,1],dtype=np.int8),per//4)
    c_template=np.tile(np.array([1,2,1,2],dtype=np.int8),per//4)
    labels=[];contexts=[];envs=[]
    for j in range(len(environments)):
        perm=rng.permutation(per);labels.append(y_template[perm]);contexts.append(c_template[perm]);envs.append(np.full(per,j,dtype=np.int64))
    shuffle=rng.permutation(n_targets)
    yy=np.concatenate(labels)[shuffle];cc=np.concatenate(contexts)[shuffle];env=np.concatenate(envs)[shuffle]
    roots=np.arange(n_targets,dtype=np.int64);sources=np.arange(n_targets,5*n_targets,dtype=np.int64);owner=np.repeat(roots,4)
    types=np.empty(4*n_targets,dtype=np.int8)
    for j in range(n_targets):types[4*j:4*j+4]=rng.permutation(np.array([1,1,-1,-1],dtype=np.int8))
    qs=np.array([q for _,q in environments],dtype=float);prob=np.where(types==1,p_invariant,qs[env[owner]])
    correct=rng.random(len(owner))<prob;eta=np.where(correct,1,-1).astype(np.int8);signal=yy[owner]*eta
    kk=np.where(types==1,cc[owner],3-cc[owner]).astype(np.int8)
    target_noise=rng.normal(0,sigma_target,(n_targets,8)).astype(np.float32)
    source_noise=rng.normal(0,sigma_source,(len(owner),8)).astype(np.float32)
    target_rel_noise=rng.normal(0,rel_sigma,(n_targets,6)).astype(np.float32)
    source_rel_noise=rng.normal(0,rel_sigma,(len(owner),6)).astype(np.float32)
    x=np.zeros((5*n_targets,16),dtype=np.float32)
    x[:n_targets,:8]=target_noise
    x[:n_targets,8:14]=np.where(cc==1,rel_amplitude,-rel_amplitude)[:,None].astype(np.float32)+target_rel_noise
    x[:n_targets,14:]=np.array([1.,0.],dtype=np.float32)
    x[n_targets:,:8]=source_noise
    for k in [1,2]:
        chosen=kk==k;block=slice(0,4) if k==1 else slice(4,8)
        x[n_targets:,:8][chosen,block]+=signal[chosen,None].astype(np.float32)
    x[n_targets:,8:14]=np.where(kk==1,rel_amplitude,-rel_amplitude)[:,None].astype(np.float32)+source_rel_noise
    x[n_targets:,14:]=np.array([0.,1.],dtype=np.float32)
    y=np.full(5*n_targets,-1,dtype=np.int64);y[:n_targets]=((yy+1)//2).astype(np.int64)
    mask=np.arange(5*n_targets)<n_targets
    graph={'x':x,'edge_index':np.stack([sources,owner]),'y':y,
           'train_mask':mask.copy() if split=='train' else np.zeros(5*n_targets,dtype=bool),
           'val_mask':mask.copy() if split=='development' else np.zeros(5*n_targets,dtype=bool),
           'test_mask':np.zeros(5*n_targets,dtype=bool)}
    meta={'target_id':roots,'target_label_sign':yy,'target_context':cc,'env_id':env,
          'environment_names':np.array([name for name,_ in environments]),'environment_spurious_probability':qs,
          'support_id':sources,'support_target_id':owner,'edge_type':types,'edge_env_id':env[owner],
          'signal_correctness':correct,'signal_eta':eta,'support_signal':signal,'support_context':kk,
          'support_expected_correctness':prob,'target_signal_noise':target_noise,'source_signal_noise':source_noise,
          'target_relation_noise':target_rel_noise,'source_relation_noise':source_rel_noise}
    return graph,meta,{'generation':{'seed':seed,'stream':stream,'p_invariant':p_invariant,
        'sigma_source':sigma_source,'sigma_target':sigma_target,'rel_amplitude':rel_amplitude,'rel_sigma':rel_sigma,
        'conditional_independence':'four independent support signs given target Y/c/q; within-node four active dimensions share T only',
        'baseline_scope':'Balanced labels and contexts with invariant and spurious incoming edges'},
        'structure':{'n_targets':n_targets,'n_nodes':5*n_targets,'nonself_edges':4*n_targets,'feature_dim':16}}

def _auc(labels,scores):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(labels,scores))
