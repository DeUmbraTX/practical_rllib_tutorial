Doc: https://docs.ray.io/en/latest/index.html

We are working with ray 1.10.0
```
conda create -n rllib python=3.8
conda activate rllib
pip install "ray[rllib]" tensorflow torch

pip install pygame  # so we can visualize what is going on, 2.1.2
```

```
cd /Users/jmugan/ray_results/YourTrainer
ls -ltr
# go to latest
tensorboard --logdir ./
```

# Credits
Icons from publicdomainvectors.org https://publicdomainvectors.org/  

robot https://publicdomainvectors.org/en/free-clipart/Yellow-robot/81372.html

chicken https://publicdomainvectors.org/en/free-clipart/Vector-illustration-of-cartoon-chicken-confused/24675.htm