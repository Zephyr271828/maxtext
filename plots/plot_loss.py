import matplotlib.pyplot as plt 


def extract_fms_loss(fpath):
    losses = []
    with open(fpath, 'r') as f:
        for l in f.readlines():
            if l.startswith('loss: '):
                loss = float(l[len('loss: '):])
                losses.append(loss)
    return losses

def extract_maxtext_loss(fpath):
    losses = []
    with open(fpath, 'r') as f:
        for l in f.readlines():
            if l.startswith('completed step: '):
                loss = float(l[l.find('loss: ') + len('loss: '):])
                losses.append(loss)
    return losses

fms_losses = extract_fms_loss('/home/zephyr/gcs-bucket/pruning/fms-grad-accum/logs/benchmarking_20250728_021251.log')
maxtext_losses = extract_maxtext_loss('/home/zephyr/maxtext/logs/training_20250728_035003.log')

plt.plot([i for i in range(len(fms_losses))], fms_losses, label='fms')
plt.plot([i for i in range(len(maxtext_losses))], maxtext_losses, label='maxtext')
plt.legend()

plt.grid(True)
plt.tight_layout()
plt.savefig('losses.png')

