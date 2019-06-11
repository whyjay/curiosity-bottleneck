## Curiosity-Bottleneck: Exploration by Distilling Task-Specific Novelty ##

Youngjin Kim, Wontae Nam*, Hyunwoo Kim*, Jihoon Kim, Gunhee Kim<br/>
&#42;equal contribution

Vision and Learning Lab., Seoul National University<br/>
Clova, Naver

#### Installation
```
pip install -r requirements
```

#### Run

The following command should train an PPO agent with Curiosity-Bottleneck on Gravitar
```bash
python run_atari.py
```

#### Acknowledgements
This code is based on [RND](https://github.com/openai/random-network-distillation) implementation by Yuri Burda

#### Citation

```
@inproceedings{
	kim2019curiositybottleneck,
	title={Curiosity-Bottleneck: Exploration by Distilling Task-Specific Novelty},
	author={Youngjin Kim and Wontae Nam and Hyunwoo Kim and Jihoon Kim and Gunhee Kim},
	booktitle={International Conference on Machine Learning},
	year={2019}
}
```

