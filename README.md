
# Implementação da arquitetura AlexNet para classificação de imagens do dataset Cifar10.






**Equipe**

Welington Henrique de Almeida Gomes

**Dataset**

Cifar10, o mesmo pode ser obtido no seguinte endereço: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz


**Descrição do projeto**

Foi realizada a implementação da arquitetura AlexNet para classificação de imagens que compõem o dataset Cifar10, o mesmo possui 60.000 imagens coloridas, 32x32, com 10 classes, sendo elas: ['Avião',  'Automóvel', 'Pássaro', 'Gato', 'Cervo', 'Cachorro',  'Sapo', ‘Cavalo',  'Navio', ‘Caminhão']. 

O mesmo é dividido em 50.000 para treinamento e 10.000 para teste. 


Inicialmente, a ideia seria implementar a CNN para o dataset Cifar10 e Cifar100, porém, no Cifar100 estava gerando overfiting, devido a baixa quantidade de amostras de treinamento e também a alta complexidade da AlexNet frente a baixa qualidade das imagens, assim sendo, optou-se pela implementação somente no Cifar10.

Após realizarmos a compiliação da rede neural convolucional e obter a acurácia, realizamos uma nova abordagem utiilizando Data Augmentation, desta forma, aumentamos nosso conjunto de treinamento e obtivemos uma acurácia e matriz confusão com melhores performance.

Na abordagem Data Augmentation, utilizamos os seguintes parâmetros para aumentarmos nossos dados de treinamento artificialmente:

1 - rotation_range: As imagens foram rotacionadas no intervalo de -20 a +20 graus, criando variações na orientação das imagens.

2 - width_shift_range e height_shift_range: As imagens foram deslocadas horizontalmente e verticalmente no intervalo de -0.2 a +0.2 da largura e altura da imagem, respectivamente. Isso criou variações de posição das imagens dentro do quadro.

3 - shear_range: As imagens sofreram transformações de cisalhamento no intervalo de -0.2 a +0.2, o que distorceu a forma da imagem.

4 - zoom_range: As imagens foram ampliadas ou reduzidas em escala no intervalo de 0.8 a 1.2. Isso criou variações de tamanho das imagens.

5 - horizontal_flip: As imagens foram espelhadas horizontalmente, ou seja, invertidas da esquerda para a direita. Essa transformação ajudou a criar variações de orientação das imagens.

6 - fill_mode: Esta configuração determinou a estratégia utilizada para preencher os pixels gerados durante as transformações anteriores. O modo 'nearest' preencheu os pixels com os valores dos pixels mais próximos.

**Repositório do projeto**

https://github.com/welingtongomes/ProjetoCifar_10

**Classificador e acurácia**

Ao realizar a primeira implementação e gerar a matriz confusão, obteve-se a matriz abaixo:

[[633  45  32  36  31  10  10  17 129  57]
 [ 21 732   4  11   3   4   7  11  44 163]
 [106  14 354 110 139  92  70  55  35  25]
 [ 28  17  52 467  64 161  99  43  30  39]
 [ 41   9  61  98 539  48  85  95  18   6]
 [ 16   8  49 280  71 430  48  57  18  23]
 [ 10  17  50 105 102  35 634  17  12  18]
 [ 21  15  34  94  89  60  16 628   2  41]
 [ 87  55  16  23  15   3   3   0 745  53]
 [ 44 157   7  22   6   7  12  16  50 679]]
	
O total de amostras corretamente previstas para todas as classes foi de: 5841 ,de um total de 10.000 imagens de teste.
A acurácia do modelo, após 30 (trinta) épocas foi de 58,41%.



A segunda implementação, ao qual foi adicionada a abordagem Data Augmentation retornou resultados melhores (8,4% de aumento na performance), obteve-se a matriz confusão abaixo:

[[616  51  43  24  14   4  17  26  95 110]
 [  1 796   1   5   0   1   4   4   9 179]
 [ 54  20 404  45 120  60 195  48  23  31]
 [ 11  47  25 414  40 188 163  44  16  52]
 [ 15  11  32  46 496  20 233 117  20  10]
 [  4  24  25 137  57 505 112  86  11  39]
 [  7  10  16  45  31  13 828  13   9  28]
 [  9  16  15  37  37  39  46 714   5  82]
 [ 34  98   6  12   3   4   5   4 752  82]
 [ 14 121   4  10   2   3   9   8  17 812]]



O total de amostras corretamente previstas para todas as classes é de: 6337 ,de um total de 10.000 imagens
A acurácia do modelo, após 30 (trinta) épocas foi de 63,37%.


**Instalação e Execução**

Para execução no google colab, basta realizar os comandos de importações à seguir:

import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns

import itertools

from keras.models import Sequential

from keras.layers.core import Dense, Dropout, Activation, Flatten

from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D

from keras.layers.normalization import batch_normalization

from keras.regularizers import l2

from keras.datasets import cifar10,cifar100

from keras.optimizers import Adam

from keras.utils.np_utils import to_categorical

from keras.engine.sequential import input_layer

from keras import regularizers

from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import confusion_matrix

Para executar localmente, faz-se necessária instalação das bibliotecas, por favor, realize os os passos abaixo:

	Abra o CMD;

	Digite: pip install numpy e aguarde a instalação;

	Digite: pip install matplotlib e aguarde a instalação;

	Digite: pip install seaborn e aguarde a instalação;

	Digite: pip install more-itertools e aguarde a instalação;

	Digite: pip install keras e aguarde a instalação;

  Digite: pip install tensorflow e aguarde a instalação;

  Digite: pip install scikit-learn e aguarde a instalação;






