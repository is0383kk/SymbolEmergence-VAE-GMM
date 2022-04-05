# SymbolEmergence-VAE-GMM
Naming game with probabilistic inference between agents represented by VAE and GMM.  

Model Overview.  
Each agent is represented by a VAE and a GMM.  
Agents reason in terms of naming probabilistic inferences based on the Metropolis Hastings algorithm.  
<div>
<img src='/image/se_vaegmm.png' width="400px">
<img src='/image/define.png' width="400px">
</div>

---
**The objective of the naming game is to match the variables (w^A, w^B) of both agents**  
Agents play the naming game in the following sequence：  
1. Estimation of latent variables (z^A, z^B) by VAE：Agents A and B estimate a latent variable (z^A, z^B) that follows a multivariate normal distribution from image observations. This is done by the VAE within the agent. 
2. Agent A is the speaker：Agent A clusters variables (z^A), estimates discrete variables (w^A) following a categorical distribution, and proposes them to Agent B.
3. Agent B is the listener：Agent B decides whether to accept or reject the variable (z^B) proposed by Agent A using the Metropolis Hastings method.
4. Swap the speaker and listener：The Metropolis Hastings algorithm is executed with Agent B as the speaker and Agent A as the listener.
5. Knowledge Update：The agent updates the parameters of the multivariate normal distribution.
The updated parameters are then redefined as the parameters of the VAE prior distribution and return to 1.  

---
What this repo contains:
- `main.py`: Main code for training model. Outputs ARI and Kappa coefficients.
- `cnn_vae_module_mnist.py`: A training program for VAE, running in main.py.
- `recall_image.py`: Recall the image to the agent.
- `tool.py`: Various functions handled in the program.

# How to run
You can train model by running `main.py`.  
```bash
# Communication (Metropolis Hastings algorithm) 
python main.py 

# No Communication 
python main.py --mode 0

# All Acceptance 
python main.py --mode 1
```
# Sample output

```
hogehoge:~/workspace$ python3 main.py 

CUDA True
Dataset : MNIST
Total data:10000, Category:10
VAE_iter:50, Batch_size:10
MH_iter:50, MH_mode:-1(-1:Com 0:No-com 1:All accept)
------------------Mutual learning session 0 begins------------------
VAE_Agent A Training Start(0): Epoch:50, Batch_size:10
====> Epoch: 1 Average loss: 168.2284
====> Epoch: 25 Average loss: 98.9265
====> Epoch: 50 Average loss: 92.6812
VAE_Agent B Training Start(0): Epoch:50, Batch_size:10
====> Epoch: 1 Average loss: 160.5718
====> Epoch: 25 Average loss: 98.2610
====> Epoch: 50 Average loss: 91.9621
M-H algorithm Start(0): Epoch:50
=> Epoch: 1, ARI_A: 0.023, ARI_B: 0.015, Kappa:0.422, A2B:5523, B2A:2410
=> Epoch: 10, ARI_A: 0.41, ARI_B: 0.387, Kappa:0.853, A2B:8585, B2A:8743
=> Epoch: 20, ARI_A: 0.665, ARI_B: 0.66, Kappa:0.915, A2B:8990, B2A:9128
=> Epoch: 30, ARI_A: 0.735, ARI_B: 0.736, Kappa:0.933, A2B:9168, B2A:9289
=> Epoch: 40, ARI_A: 0.779, ARI_B: 0.783, Kappa:0.945, A2B:9212, B2A:9284
=> Epoch: 50, ARI_A: 0.803, ARI_B: 0.808, Kappa:0.95, A2B:9278, B2A:9390
Iteration:0 Done:max_ARI_A: 0.803, max_ARI_B: 0.808, max_Kappa:0.95
```
About the evaluation index:
- `ARI_A`: ARI of Agent A：The degree of agreement between agent A's sine variable w^A and the true MNIST label．
- `ARI_B`: ARI of Agent B：The degree of agreement between agent B's sine variable w^B and the true MNIST label．
- `Kappa`: Kappa coefficients：The degree of agreement of the sine variables w^A and w^B between agents．
- `A2B`: When speaker A and listener B, the number of times B accepted the sign proposed by A.
- `B2A`: When speaker B and listener A, the number of times A accepted the sign proposed by B.

# Recalled image by Agents
Agents can recall the image after the naming game is over.  
Image recall uses the mean parameters estimated by the GMM for the latent variables in the VAE within the agent.  
This mean parameter is input to the VAE decoder to generate the image.  

After the naming game by `main.py` is finished, run `recall_image.py`.  
Recall image of Agent A in `/model/debug/reconA/`：
<div>
<img src='/image/recall_A.png' width="400px">
</div>

Recall image of Agent B in `/model/debug/reconB/`：
<div>
<img src='/image/recall_B.png' width="400px">
</div>
Communication between agents shows that the objects in the recalled image are shared.  



----
# VAEを活用した実画像からの記号創発
VAEとGMMによって表現されるエージェント間の確率的推論によるネーミングゲームにより，両エージェントの記号を共有することを目的とします．

リポジトリのプログラムについて:
- `main.py`: メインプログラム．エージェント間でネーミングゲームを行います．
- `cnn_vae_module_mnist.py`: main.py内でVAEの学習を行わせるプログラム．
- `recall_image.py`: 学習後のエージェントに画像の想起を行わせるプログラム．
- `tool.py`: 様々な関数が格納されたプログラム.

# 実行方法
`main.py`を実行することで確率的推論によるネーミングゲームを行います．
```bash
# コミュニケーションモデル：メトロポリスヘイスティングス法によるネーミングゲームを行うモデル
python main.py 

# ノンコミュニケーションモデル：個々のエージェントが独立して推論を行うモデル
python main.py --mode 0

# All Acceptance model：コミュニケーションを行いますが発話者の提案を聞き手が全て受容してしまうモデル
python main.py --mode 1
```
# 出力例
```
hogehoge:~/workspace$ python3 main.py 

CUDA True
Dataset : MNIST
Total data:10000, Category:10
VAE_iter:50, Batch_size:10
MH_iter:50, MH_mode:-1(-1:Com 0:No-com 1:All accept)
------------------Mutual learning session 0 begins------------------
VAE_Agent A Training Start(0): Epoch:50, Batch_size:10
====> Epoch: 1 Average loss: 168.2284
====> Epoch: 25 Average loss: 98.9265
====> Epoch: 50 Average loss: 92.6812
VAE_Agent B Training Start(0): Epoch:50, Batch_size:10
====> Epoch: 1 Average loss: 160.5718
====> Epoch: 25 Average loss: 98.2610
====> Epoch: 50 Average loss: 91.9621
M-H algorithm Start(0): Epoch:50
=> Epoch: 1, ARI_A: 0.023, ARI_B: 0.015, Kappa:0.422, A2B:5523, B2A:2410
=> Epoch: 10, ARI_A: 0.41, ARI_B: 0.387, Kappa:0.853, A2B:8585, B2A:8743
=> Epoch: 20, ARI_A: 0.665, ARI_B: 0.66, Kappa:0.915, A2B:8990, B2A:9128
=> Epoch: 30, ARI_A: 0.735, ARI_B: 0.736, Kappa:0.933, A2B:9168, B2A:9289
=> Epoch: 40, ARI_A: 0.779, ARI_B: 0.783, Kappa:0.945, A2B:9212, B2A:9284
=> Epoch: 50, ARI_A: 0.803, ARI_B: 0.808, Kappa:0.95, A2B:9278, B2A:9390
Iteration:0 Done:max_ARI_A: 0.803, max_ARI_B: 0.808, max_Kappa:0.95
```
評価値について:
- `ARI_A`: エージェントAのARI：エージェントAのサイン変数 w^A と真のMNISTラベルとの一致度合いを表す．
- `ARI_B`: エージェントBのARI：エージェントBのサイン変数 w^B と真のMNISTラベルとの一致度合いを表す．
- `Kappa`: カッパ係数：エージェント間のサイン変数 w^A と w^B の一致度合いを表す．
- `A2B`: 発話者A・聞き手Bのとき，Aが提案したサインをBが受容した回数.
- `B2A`: 発話者B・聞き手Aのとき，Bが提案したサインをAが受容した回数.

# エージェントによる画像の想起
ネーミングゲームの終了後にエージェントは画像の想起を行うことができます.  
エージェントによる画像の想起では，エージェント内のVAEが推論した潜在変数に対してGMMが推定した平均パラメータを用います.  
この平均パラメータをVAEデコーダに入力することで画像を再構成しエージェントに画像を想起させます.  
  
  
`main.py`によるネーミングゲーム終了後，`recall_image.py`を実行してください．

Recall image of Agent A in `/model/debug/reconA/`：
<div>
<img src='/image/recall_A.png' width="400px">
</div>

Recall image of Agent B in `/model/debug/reconB/`：
<div>
<img src='/image/recall_B.png' width="400px">
</div>
コミュニケーションを行ったモデルではエージェントの想起画像が共有されていることがわかります.  
