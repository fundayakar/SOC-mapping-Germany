"""Execute unchanged experiment partitions; summarize only after all finish."""
from pathlib import Path
import argparse,concurrent.futures as cf,subprocess,sys,time

HERE=Path(__file__).resolve().parent

def command(year,protocol='block_100',repeats=5):
    results_folder='results_buffer20' if protocol.endswith('buffer20') else 'results'
    return [sys.executable,str(HERE/'train_patch_models.py'),'--data',str(HERE/'data'/f'patches_{year}.npz'),
        '--output',str(HERE/results_folder/str(year)),'--device','cpu','--protocol',protocol,'--repeats',str(repeats),'--epochs','100','--patience','15']

def partition(year,seed,protocol,repeats):
    logs=HERE/'logs';logs.mkdir(exist_ok=True);path=logs/f'{protocol}_{year}_seed{seed}.log'
    start=time.time()
    with path.open('w') as f:
        process=subprocess.run(command(year,protocol,repeats)+['--only-seed',str(seed)],stdout=f,stderr=subprocess.STDOUT)
    if process.returncode:raise RuntimeError(f'{year} seed {seed} failed: '+path.read_text()[-3000:])
    return year,seed,time.time()-start

def main(workers,protocol,repeats):
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        fs=[pool.submit(partition,year,seed,protocol,repeats) for year in [2018,2022] for seed in range(repeats)]
        for i,f in enumerate(cf.as_completed(fs),1):
            year,seed,seconds=f.result();print(f'Completed partition {i}/{2*repeats}: {year}, seed {seed}, {seconds:.1f}s',flush=True)
    for year in [2018,2022]:
        subprocess.run(command(year,protocol,repeats),check=True)
        print(f'All {5*repeats*8} model-fold fits verified for {year} / {protocol}',flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--workers',type=int,default=5);a.add_argument('--protocol',choices=['block_100','block_100_buffer20'],default='block_100');a.add_argument('--repeats',type=int,default=None);args=a.parse_args();main(args.workers,args.protocol,args.repeats or (3 if args.protocol.endswith('buffer20') else 5))
