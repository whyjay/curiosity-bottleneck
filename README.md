# Curiosity-Bottleneck: <br/>Exploration by Distilling Task-Specific Novelty

This project hosts the code for our [ICML 2019 paper](http://proceedings.mlr.press/v97/kim19c.html). <br/>

![Model](https://github.com/skywalker023/skywalker023.github.io/blob/master/images/thumbs/icml_model.png?raw=true)



## Authors
[Youngjin Kim](http://vision.snu.ac.kr/people/youngjinkim.html), [Wontae Nam](https://www.linkedin.com/in/daniel-w-nam/)\*, [Hyunwoo Kim](https://skywalker023.github.io)\*, Jihoon Kim, [Gunhee Kim](http://vision.snu.ac.kr/~gunhee/) <sub><sup>(&#42;equal contribution)</sup></sub> <br/>

[Vision and Learning Lab.](http://vision.snu.ac.kr) @ Computer Science and Engineering, Seoul National University<br/>
[Clova](https://clova.ai/en/research/research-area-detail.html?id=0), NAVER

### Installation
```
pip install -r requirements
```

### Run

The following command should train a PPO agent with Curiosity-Bottleneck on Gravitar.
```bash
python run_atari.py
```

### Acknowledgements
This code is based on the [RND](https://github.com/openai/random-network-distillation) implementation by Yuri Burda.

### Citation

```
@inproceedings{
	kim2019curiositybottleneck,
	title={Curiosity-Bottleneck: Exploration by Distilling Task-Specific Novelty},
	author={Youngjin Kim and Wontae Nam and Hyunwoo Kim and Jihoon Kim and Gunhee Kim},
	booktitle={International Conference on Machine Learning},
	year={2019}
}
```

