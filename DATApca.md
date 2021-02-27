
이 brest cancer wisconsin data를 통해서 의사결정을 도와줄 수 있는 효과적인 classification model을 만드는 것이 목적이다

그렇기 떄문에 다양한 데이터 전처리: data transformation, feature selection 을 이용하고 다양한 decision model algorithm을 이용해 여러가지 모델들의 accuracy를 비교하고 그중 가장 성능이 좋은 모델을 채택할 것이다

우선 사용하는 데이터에 대해 간단한 설명을 하자면 





```python
import pandas as pd                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
table = pd.read_csv("wdbc.csv",header = None)
```


```python
table.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>22</th>
      <th>23</th>
      <th>24</th>
      <th>25</th>
      <th>26</th>
      <th>27</th>
      <th>28</th>
      <th>29</th>
      <th>30</th>
      <th>31</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>842302</td>
      <td>M</td>
      <td>17.99</td>
      <td>10.38</td>
      <td>122.80</td>
      <td>1001.0</td>
      <td>0.11840</td>
      <td>0.27760</td>
      <td>0.3001</td>
      <td>0.14710</td>
      <td>...</td>
      <td>25.38</td>
      <td>17.33</td>
      <td>184.60</td>
      <td>2019.0</td>
      <td>0.1622</td>
      <td>0.6656</td>
      <td>0.7119</td>
      <td>0.2654</td>
      <td>0.4601</td>
      <td>0.11890</td>
    </tr>
    <tr>
      <th>1</th>
      <td>842517</td>
      <td>M</td>
      <td>20.57</td>
      <td>17.77</td>
      <td>132.90</td>
      <td>1326.0</td>
      <td>0.08474</td>
      <td>0.07864</td>
      <td>0.0869</td>
      <td>0.07017</td>
      <td>...</td>
      <td>24.99</td>
      <td>23.41</td>
      <td>158.80</td>
      <td>1956.0</td>
      <td>0.1238</td>
      <td>0.1866</td>
      <td>0.2416</td>
      <td>0.1860</td>
      <td>0.2750</td>
      <td>0.08902</td>
    </tr>
    <tr>
      <th>2</th>
      <td>84300903</td>
      <td>M</td>
      <td>19.69</td>
      <td>21.25</td>
      <td>130.00</td>
      <td>1203.0</td>
      <td>0.10960</td>
      <td>0.15990</td>
      <td>0.1974</td>
      <td>0.12790</td>
      <td>...</td>
      <td>23.57</td>
      <td>25.53</td>
      <td>152.50</td>
      <td>1709.0</td>
      <td>0.1444</td>
      <td>0.4245</td>
      <td>0.4504</td>
      <td>0.2430</td>
      <td>0.3613</td>
      <td>0.08758</td>
    </tr>
    <tr>
      <th>3</th>
      <td>84348301</td>
      <td>M</td>
      <td>11.42</td>
      <td>20.38</td>
      <td>77.58</td>
      <td>386.1</td>
      <td>0.14250</td>
      <td>0.28390</td>
      <td>0.2414</td>
      <td>0.10520</td>
      <td>...</td>
      <td>14.91</td>
      <td>26.50</td>
      <td>98.87</td>
      <td>567.7</td>
      <td>0.2098</td>
      <td>0.8663</td>
      <td>0.6869</td>
      <td>0.2575</td>
      <td>0.6638</td>
      <td>0.17300</td>
    </tr>
    <tr>
      <th>4</th>
      <td>84358402</td>
      <td>M</td>
      <td>20.29</td>
      <td>14.34</td>
      <td>135.10</td>
      <td>1297.0</td>
      <td>0.10030</td>
      <td>0.13280</td>
      <td>0.1980</td>
      <td>0.10430</td>
      <td>...</td>
      <td>22.54</td>
      <td>16.67</td>
      <td>152.20</td>
      <td>1575.0</td>
      <td>0.1374</td>
      <td>0.2050</td>
      <td>0.4000</td>
      <td>0.1625</td>
      <td>0.2364</td>
      <td>0.07678</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 32 columns</p>
</div>




```python
name =["ID", "diagnosis","radius","texture","perimeter","area","smoothness","compactness","concavity","concave points","symmetry","fractal dimention",
"radius_SD","texture_SD","perimeter_SD","area_SD","smoothness_SD","compactness_SD","concavity_SD","concave points_SD","symmetry_SD","fractal dimention_SD",
"radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst","concave points_worst","symmetry_worst","fractal dimention_worst"
 ]
table.columns = name
```


```python
table
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>diagnosis</th>
      <th>radius</th>
      <th>texture</th>
      <th>perimeter</th>
      <th>area</th>
      <th>smoothness</th>
      <th>compactness</th>
      <th>concavity</th>
      <th>concave points</th>
      <th>...</th>
      <th>radius_worst</th>
      <th>texture_worst</th>
      <th>perimeter_worst</th>
      <th>area_worst</th>
      <th>smoothness_worst</th>
      <th>compactness_worst</th>
      <th>concavity_worst</th>
      <th>concave points_worst</th>
      <th>symmetry_worst</th>
      <th>fractal dimention_worst</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>842302</td>
      <td>M</td>
      <td>17.990</td>
      <td>10.38</td>
      <td>122.80</td>
      <td>1001.0</td>
      <td>0.11840</td>
      <td>0.27760</td>
      <td>0.300100</td>
      <td>0.147100</td>
      <td>...</td>
      <td>25.380</td>
      <td>17.33</td>
      <td>184.60</td>
      <td>2019.0</td>
      <td>0.16220</td>
      <td>0.66560</td>
      <td>0.71190</td>
      <td>0.26540</td>
      <td>0.4601</td>
      <td>0.11890</td>
    </tr>
    <tr>
      <th>1</th>
      <td>842517</td>
      <td>M</td>
      <td>20.570</td>
      <td>17.77</td>
      <td>132.90</td>
      <td>1326.0</td>
      <td>0.08474</td>
      <td>0.07864</td>
      <td>0.086900</td>
      <td>0.070170</td>
      <td>...</td>
      <td>24.990</td>
      <td>23.41</td>
      <td>158.80</td>
      <td>1956.0</td>
      <td>0.12380</td>
      <td>0.18660</td>
      <td>0.24160</td>
      <td>0.18600</td>
      <td>0.2750</td>
      <td>0.08902</td>
    </tr>
    <tr>
      <th>2</th>
      <td>84300903</td>
      <td>M</td>
      <td>19.690</td>
      <td>21.25</td>
      <td>130.00</td>
      <td>1203.0</td>
      <td>0.10960</td>
      <td>0.15990</td>
      <td>0.197400</td>
      <td>0.127900</td>
      <td>...</td>
      <td>23.570</td>
      <td>25.53</td>
      <td>152.50</td>
      <td>1709.0</td>
      <td>0.14440</td>
      <td>0.42450</td>
      <td>0.45040</td>
      <td>0.24300</td>
      <td>0.3613</td>
      <td>0.08758</td>
    </tr>
    <tr>
      <th>3</th>
      <td>84348301</td>
      <td>M</td>
      <td>11.420</td>
      <td>20.38</td>
      <td>77.58</td>
      <td>386.1</td>
      <td>0.14250</td>
      <td>0.28390</td>
      <td>0.241400</td>
      <td>0.105200</td>
      <td>...</td>
      <td>14.910</td>
      <td>26.50</td>
      <td>98.87</td>
      <td>567.7</td>
      <td>0.20980</td>
      <td>0.86630</td>
      <td>0.68690</td>
      <td>0.25750</td>
      <td>0.6638</td>
      <td>0.17300</td>
    </tr>
    <tr>
      <th>4</th>
      <td>84358402</td>
      <td>M</td>
      <td>20.290</td>
      <td>14.34</td>
      <td>135.10</td>
      <td>1297.0</td>
      <td>0.10030</td>
      <td>0.13280</td>
      <td>0.198000</td>
      <td>0.104300</td>
      <td>...</td>
      <td>22.540</td>
      <td>16.67</td>
      <td>152.20</td>
      <td>1575.0</td>
      <td>0.13740</td>
      <td>0.20500</td>
      <td>0.40000</td>
      <td>0.16250</td>
      <td>0.2364</td>
      <td>0.07678</td>
    </tr>
    <tr>
      <th>5</th>
      <td>843786</td>
      <td>M</td>
      <td>12.450</td>
      <td>15.70</td>
      <td>82.57</td>
      <td>477.1</td>
      <td>0.12780</td>
      <td>0.17000</td>
      <td>0.157800</td>
      <td>0.080890</td>
      <td>...</td>
      <td>15.470</td>
      <td>23.75</td>
      <td>103.40</td>
      <td>741.6</td>
      <td>0.17910</td>
      <td>0.52490</td>
      <td>0.53550</td>
      <td>0.17410</td>
      <td>0.3985</td>
      <td>0.12440</td>
    </tr>
    <tr>
      <th>6</th>
      <td>844359</td>
      <td>M</td>
      <td>18.250</td>
      <td>19.98</td>
      <td>119.60</td>
      <td>1040.0</td>
      <td>0.09463</td>
      <td>0.10900</td>
      <td>0.112700</td>
      <td>0.074000</td>
      <td>...</td>
      <td>22.880</td>
      <td>27.66</td>
      <td>153.20</td>
      <td>1606.0</td>
      <td>0.14420</td>
      <td>0.25760</td>
      <td>0.37840</td>
      <td>0.19320</td>
      <td>0.3063</td>
      <td>0.08368</td>
    </tr>
    <tr>
      <th>7</th>
      <td>84458202</td>
      <td>M</td>
      <td>13.710</td>
      <td>20.83</td>
      <td>90.20</td>
      <td>577.9</td>
      <td>0.11890</td>
      <td>0.16450</td>
      <td>0.093660</td>
      <td>0.059850</td>
      <td>...</td>
      <td>17.060</td>
      <td>28.14</td>
      <td>110.60</td>
      <td>897.0</td>
      <td>0.16540</td>
      <td>0.36820</td>
      <td>0.26780</td>
      <td>0.15560</td>
      <td>0.3196</td>
      <td>0.11510</td>
    </tr>
    <tr>
      <th>8</th>
      <td>844981</td>
      <td>M</td>
      <td>13.000</td>
      <td>21.82</td>
      <td>87.50</td>
      <td>519.8</td>
      <td>0.12730</td>
      <td>0.19320</td>
      <td>0.185900</td>
      <td>0.093530</td>
      <td>...</td>
      <td>15.490</td>
      <td>30.73</td>
      <td>106.20</td>
      <td>739.3</td>
      <td>0.17030</td>
      <td>0.54010</td>
      <td>0.53900</td>
      <td>0.20600</td>
      <td>0.4378</td>
      <td>0.10720</td>
    </tr>
    <tr>
      <th>9</th>
      <td>84501001</td>
      <td>M</td>
      <td>12.460</td>
      <td>24.04</td>
      <td>83.97</td>
      <td>475.9</td>
      <td>0.11860</td>
      <td>0.23960</td>
      <td>0.227300</td>
      <td>0.085430</td>
      <td>...</td>
      <td>15.090</td>
      <td>40.68</td>
      <td>97.65</td>
      <td>711.4</td>
      <td>0.18530</td>
      <td>1.05800</td>
      <td>1.10500</td>
      <td>0.22100</td>
      <td>0.4366</td>
      <td>0.20750</td>
    </tr>
    <tr>
      <th>10</th>
      <td>845636</td>
      <td>M</td>
      <td>16.020</td>
      <td>23.24</td>
      <td>102.70</td>
      <td>797.8</td>
      <td>0.08206</td>
      <td>0.06669</td>
      <td>0.032990</td>
      <td>0.033230</td>
      <td>...</td>
      <td>19.190</td>
      <td>33.88</td>
      <td>123.80</td>
      <td>1150.0</td>
      <td>0.11810</td>
      <td>0.15510</td>
      <td>0.14590</td>
      <td>0.09975</td>
      <td>0.2948</td>
      <td>0.08452</td>
    </tr>
    <tr>
      <th>11</th>
      <td>84610002</td>
      <td>M</td>
      <td>15.780</td>
      <td>17.89</td>
      <td>103.60</td>
      <td>781.0</td>
      <td>0.09710</td>
      <td>0.12920</td>
      <td>0.099540</td>
      <td>0.066060</td>
      <td>...</td>
      <td>20.420</td>
      <td>27.28</td>
      <td>136.50</td>
      <td>1299.0</td>
      <td>0.13960</td>
      <td>0.56090</td>
      <td>0.39650</td>
      <td>0.18100</td>
      <td>0.3792</td>
      <td>0.10480</td>
    </tr>
    <tr>
      <th>12</th>
      <td>846226</td>
      <td>M</td>
      <td>19.170</td>
      <td>24.80</td>
      <td>132.40</td>
      <td>1123.0</td>
      <td>0.09740</td>
      <td>0.24580</td>
      <td>0.206500</td>
      <td>0.111800</td>
      <td>...</td>
      <td>20.960</td>
      <td>29.94</td>
      <td>151.70</td>
      <td>1332.0</td>
      <td>0.10370</td>
      <td>0.39030</td>
      <td>0.36390</td>
      <td>0.17670</td>
      <td>0.3176</td>
      <td>0.10230</td>
    </tr>
    <tr>
      <th>13</th>
      <td>846381</td>
      <td>M</td>
      <td>15.850</td>
      <td>23.95</td>
      <td>103.70</td>
      <td>782.7</td>
      <td>0.08401</td>
      <td>0.10020</td>
      <td>0.099380</td>
      <td>0.053640</td>
      <td>...</td>
      <td>16.840</td>
      <td>27.66</td>
      <td>112.00</td>
      <td>876.5</td>
      <td>0.11310</td>
      <td>0.19240</td>
      <td>0.23220</td>
      <td>0.11190</td>
      <td>0.2809</td>
      <td>0.06287</td>
    </tr>
    <tr>
      <th>14</th>
      <td>84667401</td>
      <td>M</td>
      <td>13.730</td>
      <td>22.61</td>
      <td>93.60</td>
      <td>578.3</td>
      <td>0.11310</td>
      <td>0.22930</td>
      <td>0.212800</td>
      <td>0.080250</td>
      <td>...</td>
      <td>15.030</td>
      <td>32.01</td>
      <td>108.80</td>
      <td>697.7</td>
      <td>0.16510</td>
      <td>0.77250</td>
      <td>0.69430</td>
      <td>0.22080</td>
      <td>0.3596</td>
      <td>0.14310</td>
    </tr>
    <tr>
      <th>15</th>
      <td>84799002</td>
      <td>M</td>
      <td>14.540</td>
      <td>27.54</td>
      <td>96.73</td>
      <td>658.8</td>
      <td>0.11390</td>
      <td>0.15950</td>
      <td>0.163900</td>
      <td>0.073640</td>
      <td>...</td>
      <td>17.460</td>
      <td>37.13</td>
      <td>124.10</td>
      <td>943.2</td>
      <td>0.16780</td>
      <td>0.65770</td>
      <td>0.70260</td>
      <td>0.17120</td>
      <td>0.4218</td>
      <td>0.13410</td>
    </tr>
    <tr>
      <th>16</th>
      <td>848406</td>
      <td>M</td>
      <td>14.680</td>
      <td>20.13</td>
      <td>94.74</td>
      <td>684.5</td>
      <td>0.09867</td>
      <td>0.07200</td>
      <td>0.073950</td>
      <td>0.052590</td>
      <td>...</td>
      <td>19.070</td>
      <td>30.88</td>
      <td>123.40</td>
      <td>1138.0</td>
      <td>0.14640</td>
      <td>0.18710</td>
      <td>0.29140</td>
      <td>0.16090</td>
      <td>0.3029</td>
      <td>0.08216</td>
    </tr>
    <tr>
      <th>17</th>
      <td>84862001</td>
      <td>M</td>
      <td>16.130</td>
      <td>20.68</td>
      <td>108.10</td>
      <td>798.8</td>
      <td>0.11700</td>
      <td>0.20220</td>
      <td>0.172200</td>
      <td>0.102800</td>
      <td>...</td>
      <td>20.960</td>
      <td>31.48</td>
      <td>136.80</td>
      <td>1315.0</td>
      <td>0.17890</td>
      <td>0.42330</td>
      <td>0.47840</td>
      <td>0.20730</td>
      <td>0.3706</td>
      <td>0.11420</td>
    </tr>
    <tr>
      <th>18</th>
      <td>849014</td>
      <td>M</td>
      <td>19.810</td>
      <td>22.15</td>
      <td>130.00</td>
      <td>1260.0</td>
      <td>0.09831</td>
      <td>0.10270</td>
      <td>0.147900</td>
      <td>0.094980</td>
      <td>...</td>
      <td>27.320</td>
      <td>30.88</td>
      <td>186.80</td>
      <td>2398.0</td>
      <td>0.15120</td>
      <td>0.31500</td>
      <td>0.53720</td>
      <td>0.23880</td>
      <td>0.2768</td>
      <td>0.07615</td>
    </tr>
    <tr>
      <th>19</th>
      <td>8510426</td>
      <td>B</td>
      <td>13.540</td>
      <td>14.36</td>
      <td>87.46</td>
      <td>566.3</td>
      <td>0.09779</td>
      <td>0.08129</td>
      <td>0.066640</td>
      <td>0.047810</td>
      <td>...</td>
      <td>15.110</td>
      <td>19.26</td>
      <td>99.70</td>
      <td>711.2</td>
      <td>0.14400</td>
      <td>0.17730</td>
      <td>0.23900</td>
      <td>0.12880</td>
      <td>0.2977</td>
      <td>0.07259</td>
    </tr>
    <tr>
      <th>20</th>
      <td>8510653</td>
      <td>B</td>
      <td>13.080</td>
      <td>15.71</td>
      <td>85.63</td>
      <td>520.0</td>
      <td>0.10750</td>
      <td>0.12700</td>
      <td>0.045680</td>
      <td>0.031100</td>
      <td>...</td>
      <td>14.500</td>
      <td>20.49</td>
      <td>96.09</td>
      <td>630.5</td>
      <td>0.13120</td>
      <td>0.27760</td>
      <td>0.18900</td>
      <td>0.07283</td>
      <td>0.3184</td>
      <td>0.08183</td>
    </tr>
    <tr>
      <th>21</th>
      <td>8510824</td>
      <td>B</td>
      <td>9.504</td>
      <td>12.44</td>
      <td>60.34</td>
      <td>273.9</td>
      <td>0.10240</td>
      <td>0.06492</td>
      <td>0.029560</td>
      <td>0.020760</td>
      <td>...</td>
      <td>10.230</td>
      <td>15.66</td>
      <td>65.13</td>
      <td>314.9</td>
      <td>0.13240</td>
      <td>0.11480</td>
      <td>0.08867</td>
      <td>0.06227</td>
      <td>0.2450</td>
      <td>0.07773</td>
    </tr>
    <tr>
      <th>22</th>
      <td>8511133</td>
      <td>M</td>
      <td>15.340</td>
      <td>14.26</td>
      <td>102.50</td>
      <td>704.4</td>
      <td>0.10730</td>
      <td>0.21350</td>
      <td>0.207700</td>
      <td>0.097560</td>
      <td>...</td>
      <td>18.070</td>
      <td>19.08</td>
      <td>125.10</td>
      <td>980.9</td>
      <td>0.13900</td>
      <td>0.59540</td>
      <td>0.63050</td>
      <td>0.23930</td>
      <td>0.4667</td>
      <td>0.09946</td>
    </tr>
    <tr>
      <th>23</th>
      <td>851509</td>
      <td>M</td>
      <td>21.160</td>
      <td>23.04</td>
      <td>137.20</td>
      <td>1404.0</td>
      <td>0.09428</td>
      <td>0.10220</td>
      <td>0.109700</td>
      <td>0.086320</td>
      <td>...</td>
      <td>29.170</td>
      <td>35.59</td>
      <td>188.00</td>
      <td>2615.0</td>
      <td>0.14010</td>
      <td>0.26000</td>
      <td>0.31550</td>
      <td>0.20090</td>
      <td>0.2822</td>
      <td>0.07526</td>
    </tr>
    <tr>
      <th>24</th>
      <td>852552</td>
      <td>M</td>
      <td>16.650</td>
      <td>21.38</td>
      <td>110.00</td>
      <td>904.6</td>
      <td>0.11210</td>
      <td>0.14570</td>
      <td>0.152500</td>
      <td>0.091700</td>
      <td>...</td>
      <td>26.460</td>
      <td>31.56</td>
      <td>177.00</td>
      <td>2215.0</td>
      <td>0.18050</td>
      <td>0.35780</td>
      <td>0.46950</td>
      <td>0.20950</td>
      <td>0.3613</td>
      <td>0.09564</td>
    </tr>
    <tr>
      <th>25</th>
      <td>852631</td>
      <td>M</td>
      <td>17.140</td>
      <td>16.40</td>
      <td>116.00</td>
      <td>912.7</td>
      <td>0.11860</td>
      <td>0.22760</td>
      <td>0.222900</td>
      <td>0.140100</td>
      <td>...</td>
      <td>22.250</td>
      <td>21.40</td>
      <td>152.40</td>
      <td>1461.0</td>
      <td>0.15450</td>
      <td>0.39490</td>
      <td>0.38530</td>
      <td>0.25500</td>
      <td>0.4066</td>
      <td>0.10590</td>
    </tr>
    <tr>
      <th>26</th>
      <td>852763</td>
      <td>M</td>
      <td>14.580</td>
      <td>21.53</td>
      <td>97.41</td>
      <td>644.8</td>
      <td>0.10540</td>
      <td>0.18680</td>
      <td>0.142500</td>
      <td>0.087830</td>
      <td>...</td>
      <td>17.620</td>
      <td>33.21</td>
      <td>122.40</td>
      <td>896.9</td>
      <td>0.15250</td>
      <td>0.66430</td>
      <td>0.55390</td>
      <td>0.27010</td>
      <td>0.4264</td>
      <td>0.12750</td>
    </tr>
    <tr>
      <th>27</th>
      <td>852781</td>
      <td>M</td>
      <td>18.610</td>
      <td>20.25</td>
      <td>122.10</td>
      <td>1094.0</td>
      <td>0.09440</td>
      <td>0.10660</td>
      <td>0.149000</td>
      <td>0.077310</td>
      <td>...</td>
      <td>21.310</td>
      <td>27.26</td>
      <td>139.90</td>
      <td>1403.0</td>
      <td>0.13380</td>
      <td>0.21170</td>
      <td>0.34460</td>
      <td>0.14900</td>
      <td>0.2341</td>
      <td>0.07421</td>
    </tr>
    <tr>
      <th>28</th>
      <td>852973</td>
      <td>M</td>
      <td>15.300</td>
      <td>25.27</td>
      <td>102.40</td>
      <td>732.4</td>
      <td>0.10820</td>
      <td>0.16970</td>
      <td>0.168300</td>
      <td>0.087510</td>
      <td>...</td>
      <td>20.270</td>
      <td>36.71</td>
      <td>149.30</td>
      <td>1269.0</td>
      <td>0.16410</td>
      <td>0.61100</td>
      <td>0.63350</td>
      <td>0.20240</td>
      <td>0.4027</td>
      <td>0.09876</td>
    </tr>
    <tr>
      <th>29</th>
      <td>853201</td>
      <td>M</td>
      <td>17.570</td>
      <td>15.05</td>
      <td>115.00</td>
      <td>955.1</td>
      <td>0.09847</td>
      <td>0.11570</td>
      <td>0.098750</td>
      <td>0.079530</td>
      <td>...</td>
      <td>20.010</td>
      <td>19.52</td>
      <td>134.90</td>
      <td>1227.0</td>
      <td>0.12550</td>
      <td>0.28120</td>
      <td>0.24890</td>
      <td>0.14560</td>
      <td>0.2756</td>
      <td>0.07919</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>539</th>
      <td>921362</td>
      <td>B</td>
      <td>7.691</td>
      <td>25.44</td>
      <td>48.34</td>
      <td>170.4</td>
      <td>0.08668</td>
      <td>0.11990</td>
      <td>0.092520</td>
      <td>0.013640</td>
      <td>...</td>
      <td>8.678</td>
      <td>31.89</td>
      <td>54.49</td>
      <td>223.6</td>
      <td>0.15960</td>
      <td>0.30640</td>
      <td>0.33930</td>
      <td>0.05000</td>
      <td>0.2790</td>
      <td>0.10660</td>
    </tr>
    <tr>
      <th>540</th>
      <td>921385</td>
      <td>B</td>
      <td>11.540</td>
      <td>14.44</td>
      <td>74.65</td>
      <td>402.9</td>
      <td>0.09984</td>
      <td>0.11200</td>
      <td>0.067370</td>
      <td>0.025940</td>
      <td>...</td>
      <td>12.260</td>
      <td>19.68</td>
      <td>78.78</td>
      <td>457.8</td>
      <td>0.13450</td>
      <td>0.21180</td>
      <td>0.17970</td>
      <td>0.06918</td>
      <td>0.2329</td>
      <td>0.08134</td>
    </tr>
    <tr>
      <th>541</th>
      <td>921386</td>
      <td>B</td>
      <td>14.470</td>
      <td>24.99</td>
      <td>95.81</td>
      <td>656.4</td>
      <td>0.08837</td>
      <td>0.12300</td>
      <td>0.100900</td>
      <td>0.038900</td>
      <td>...</td>
      <td>16.220</td>
      <td>31.73</td>
      <td>113.50</td>
      <td>808.9</td>
      <td>0.13400</td>
      <td>0.42020</td>
      <td>0.40400</td>
      <td>0.12050</td>
      <td>0.3187</td>
      <td>0.10230</td>
    </tr>
    <tr>
      <th>542</th>
      <td>921644</td>
      <td>B</td>
      <td>14.740</td>
      <td>25.42</td>
      <td>94.70</td>
      <td>668.6</td>
      <td>0.08275</td>
      <td>0.07214</td>
      <td>0.041050</td>
      <td>0.030270</td>
      <td>...</td>
      <td>16.510</td>
      <td>32.29</td>
      <td>107.40</td>
      <td>826.4</td>
      <td>0.10600</td>
      <td>0.13760</td>
      <td>0.16110</td>
      <td>0.10950</td>
      <td>0.2722</td>
      <td>0.06956</td>
    </tr>
    <tr>
      <th>543</th>
      <td>922296</td>
      <td>B</td>
      <td>13.210</td>
      <td>28.06</td>
      <td>84.88</td>
      <td>538.4</td>
      <td>0.08671</td>
      <td>0.06877</td>
      <td>0.029870</td>
      <td>0.032750</td>
      <td>...</td>
      <td>14.370</td>
      <td>37.17</td>
      <td>92.48</td>
      <td>629.6</td>
      <td>0.10720</td>
      <td>0.13810</td>
      <td>0.10620</td>
      <td>0.07958</td>
      <td>0.2473</td>
      <td>0.06443</td>
    </tr>
    <tr>
      <th>544</th>
      <td>922297</td>
      <td>B</td>
      <td>13.870</td>
      <td>20.70</td>
      <td>89.77</td>
      <td>584.8</td>
      <td>0.09578</td>
      <td>0.10180</td>
      <td>0.036880</td>
      <td>0.023690</td>
      <td>...</td>
      <td>15.050</td>
      <td>24.75</td>
      <td>99.17</td>
      <td>688.6</td>
      <td>0.12640</td>
      <td>0.20370</td>
      <td>0.13770</td>
      <td>0.06845</td>
      <td>0.2249</td>
      <td>0.08492</td>
    </tr>
    <tr>
      <th>545</th>
      <td>922576</td>
      <td>B</td>
      <td>13.620</td>
      <td>23.23</td>
      <td>87.19</td>
      <td>573.2</td>
      <td>0.09246</td>
      <td>0.06747</td>
      <td>0.029740</td>
      <td>0.024430</td>
      <td>...</td>
      <td>15.350</td>
      <td>29.09</td>
      <td>97.58</td>
      <td>729.8</td>
      <td>0.12160</td>
      <td>0.15170</td>
      <td>0.10490</td>
      <td>0.07174</td>
      <td>0.2642</td>
      <td>0.06953</td>
    </tr>
    <tr>
      <th>546</th>
      <td>922577</td>
      <td>B</td>
      <td>10.320</td>
      <td>16.35</td>
      <td>65.31</td>
      <td>324.9</td>
      <td>0.09434</td>
      <td>0.04994</td>
      <td>0.010120</td>
      <td>0.005495</td>
      <td>...</td>
      <td>11.250</td>
      <td>21.77</td>
      <td>71.12</td>
      <td>384.9</td>
      <td>0.12850</td>
      <td>0.08842</td>
      <td>0.04384</td>
      <td>0.02381</td>
      <td>0.2681</td>
      <td>0.07399</td>
    </tr>
    <tr>
      <th>547</th>
      <td>922840</td>
      <td>B</td>
      <td>10.260</td>
      <td>16.58</td>
      <td>65.85</td>
      <td>320.8</td>
      <td>0.08877</td>
      <td>0.08066</td>
      <td>0.043580</td>
      <td>0.024380</td>
      <td>...</td>
      <td>10.830</td>
      <td>22.04</td>
      <td>71.08</td>
      <td>357.4</td>
      <td>0.14610</td>
      <td>0.22460</td>
      <td>0.17830</td>
      <td>0.08333</td>
      <td>0.2691</td>
      <td>0.09479</td>
    </tr>
    <tr>
      <th>548</th>
      <td>923169</td>
      <td>B</td>
      <td>9.683</td>
      <td>19.34</td>
      <td>61.05</td>
      <td>285.7</td>
      <td>0.08491</td>
      <td>0.05030</td>
      <td>0.023370</td>
      <td>0.009615</td>
      <td>...</td>
      <td>10.930</td>
      <td>25.59</td>
      <td>69.10</td>
      <td>364.2</td>
      <td>0.11990</td>
      <td>0.09546</td>
      <td>0.09350</td>
      <td>0.03846</td>
      <td>0.2552</td>
      <td>0.07920</td>
    </tr>
    <tr>
      <th>549</th>
      <td>923465</td>
      <td>B</td>
      <td>10.820</td>
      <td>24.21</td>
      <td>68.89</td>
      <td>361.6</td>
      <td>0.08192</td>
      <td>0.06602</td>
      <td>0.015480</td>
      <td>0.008160</td>
      <td>...</td>
      <td>13.030</td>
      <td>31.45</td>
      <td>83.90</td>
      <td>505.6</td>
      <td>0.12040</td>
      <td>0.16330</td>
      <td>0.06194</td>
      <td>0.03264</td>
      <td>0.3059</td>
      <td>0.07626</td>
    </tr>
    <tr>
      <th>550</th>
      <td>923748</td>
      <td>B</td>
      <td>10.860</td>
      <td>21.48</td>
      <td>68.51</td>
      <td>360.5</td>
      <td>0.07431</td>
      <td>0.04227</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>11.660</td>
      <td>24.77</td>
      <td>74.08</td>
      <td>412.3</td>
      <td>0.10010</td>
      <td>0.07348</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.2458</td>
      <td>0.06592</td>
    </tr>
    <tr>
      <th>551</th>
      <td>923780</td>
      <td>B</td>
      <td>11.130</td>
      <td>22.44</td>
      <td>71.49</td>
      <td>378.4</td>
      <td>0.09566</td>
      <td>0.08194</td>
      <td>0.048240</td>
      <td>0.022570</td>
      <td>...</td>
      <td>12.020</td>
      <td>28.26</td>
      <td>77.80</td>
      <td>436.6</td>
      <td>0.10870</td>
      <td>0.17820</td>
      <td>0.15640</td>
      <td>0.06413</td>
      <td>0.3169</td>
      <td>0.08032</td>
    </tr>
    <tr>
      <th>552</th>
      <td>924084</td>
      <td>B</td>
      <td>12.770</td>
      <td>29.43</td>
      <td>81.35</td>
      <td>507.9</td>
      <td>0.08276</td>
      <td>0.04234</td>
      <td>0.019970</td>
      <td>0.014990</td>
      <td>...</td>
      <td>13.870</td>
      <td>36.00</td>
      <td>88.10</td>
      <td>594.7</td>
      <td>0.12340</td>
      <td>0.10640</td>
      <td>0.08653</td>
      <td>0.06498</td>
      <td>0.2407</td>
      <td>0.06484</td>
    </tr>
    <tr>
      <th>553</th>
      <td>924342</td>
      <td>B</td>
      <td>9.333</td>
      <td>21.94</td>
      <td>59.01</td>
      <td>264.0</td>
      <td>0.09240</td>
      <td>0.05605</td>
      <td>0.039960</td>
      <td>0.012820</td>
      <td>...</td>
      <td>9.845</td>
      <td>25.05</td>
      <td>62.86</td>
      <td>295.8</td>
      <td>0.11030</td>
      <td>0.08298</td>
      <td>0.07993</td>
      <td>0.02564</td>
      <td>0.2435</td>
      <td>0.07393</td>
    </tr>
    <tr>
      <th>554</th>
      <td>924632</td>
      <td>B</td>
      <td>12.880</td>
      <td>28.92</td>
      <td>82.50</td>
      <td>514.3</td>
      <td>0.08123</td>
      <td>0.05824</td>
      <td>0.061950</td>
      <td>0.023430</td>
      <td>...</td>
      <td>13.890</td>
      <td>35.74</td>
      <td>88.84</td>
      <td>595.7</td>
      <td>0.12270</td>
      <td>0.16200</td>
      <td>0.24390</td>
      <td>0.06493</td>
      <td>0.2372</td>
      <td>0.07242</td>
    </tr>
    <tr>
      <th>555</th>
      <td>924934</td>
      <td>B</td>
      <td>10.290</td>
      <td>27.61</td>
      <td>65.67</td>
      <td>321.4</td>
      <td>0.09030</td>
      <td>0.07658</td>
      <td>0.059990</td>
      <td>0.027380</td>
      <td>...</td>
      <td>10.840</td>
      <td>34.91</td>
      <td>69.57</td>
      <td>357.6</td>
      <td>0.13840</td>
      <td>0.17100</td>
      <td>0.20000</td>
      <td>0.09127</td>
      <td>0.2226</td>
      <td>0.08283</td>
    </tr>
    <tr>
      <th>556</th>
      <td>924964</td>
      <td>B</td>
      <td>10.160</td>
      <td>19.59</td>
      <td>64.73</td>
      <td>311.7</td>
      <td>0.10030</td>
      <td>0.07504</td>
      <td>0.005025</td>
      <td>0.011160</td>
      <td>...</td>
      <td>10.650</td>
      <td>22.88</td>
      <td>67.88</td>
      <td>347.3</td>
      <td>0.12650</td>
      <td>0.12000</td>
      <td>0.01005</td>
      <td>0.02232</td>
      <td>0.2262</td>
      <td>0.06742</td>
    </tr>
    <tr>
      <th>557</th>
      <td>925236</td>
      <td>B</td>
      <td>9.423</td>
      <td>27.88</td>
      <td>59.26</td>
      <td>271.3</td>
      <td>0.08123</td>
      <td>0.04971</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>10.490</td>
      <td>34.24</td>
      <td>66.50</td>
      <td>330.6</td>
      <td>0.10730</td>
      <td>0.07158</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.2475</td>
      <td>0.06969</td>
    </tr>
    <tr>
      <th>558</th>
      <td>925277</td>
      <td>B</td>
      <td>14.590</td>
      <td>22.68</td>
      <td>96.39</td>
      <td>657.1</td>
      <td>0.08473</td>
      <td>0.13300</td>
      <td>0.102900</td>
      <td>0.037360</td>
      <td>...</td>
      <td>15.480</td>
      <td>27.27</td>
      <td>105.90</td>
      <td>733.5</td>
      <td>0.10260</td>
      <td>0.31710</td>
      <td>0.36620</td>
      <td>0.11050</td>
      <td>0.2258</td>
      <td>0.08004</td>
    </tr>
    <tr>
      <th>559</th>
      <td>925291</td>
      <td>B</td>
      <td>11.510</td>
      <td>23.93</td>
      <td>74.52</td>
      <td>403.5</td>
      <td>0.09261</td>
      <td>0.10210</td>
      <td>0.111200</td>
      <td>0.041050</td>
      <td>...</td>
      <td>12.480</td>
      <td>37.16</td>
      <td>82.28</td>
      <td>474.2</td>
      <td>0.12980</td>
      <td>0.25170</td>
      <td>0.36300</td>
      <td>0.09653</td>
      <td>0.2112</td>
      <td>0.08732</td>
    </tr>
    <tr>
      <th>560</th>
      <td>925292</td>
      <td>B</td>
      <td>14.050</td>
      <td>27.15</td>
      <td>91.38</td>
      <td>600.4</td>
      <td>0.09929</td>
      <td>0.11260</td>
      <td>0.044620</td>
      <td>0.043040</td>
      <td>...</td>
      <td>15.300</td>
      <td>33.17</td>
      <td>100.20</td>
      <td>706.7</td>
      <td>0.12410</td>
      <td>0.22640</td>
      <td>0.13260</td>
      <td>0.10480</td>
      <td>0.2250</td>
      <td>0.08321</td>
    </tr>
    <tr>
      <th>561</th>
      <td>925311</td>
      <td>B</td>
      <td>11.200</td>
      <td>29.37</td>
      <td>70.67</td>
      <td>386.0</td>
      <td>0.07449</td>
      <td>0.03558</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>11.920</td>
      <td>38.30</td>
      <td>75.19</td>
      <td>439.6</td>
      <td>0.09267</td>
      <td>0.05494</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.1566</td>
      <td>0.05905</td>
    </tr>
    <tr>
      <th>562</th>
      <td>925622</td>
      <td>M</td>
      <td>15.220</td>
      <td>30.62</td>
      <td>103.40</td>
      <td>716.9</td>
      <td>0.10480</td>
      <td>0.20870</td>
      <td>0.255000</td>
      <td>0.094290</td>
      <td>...</td>
      <td>17.520</td>
      <td>42.79</td>
      <td>128.70</td>
      <td>915.0</td>
      <td>0.14170</td>
      <td>0.79170</td>
      <td>1.17000</td>
      <td>0.23560</td>
      <td>0.4089</td>
      <td>0.14090</td>
    </tr>
    <tr>
      <th>563</th>
      <td>926125</td>
      <td>M</td>
      <td>20.920</td>
      <td>25.09</td>
      <td>143.00</td>
      <td>1347.0</td>
      <td>0.10990</td>
      <td>0.22360</td>
      <td>0.317400</td>
      <td>0.147400</td>
      <td>...</td>
      <td>24.290</td>
      <td>29.41</td>
      <td>179.10</td>
      <td>1819.0</td>
      <td>0.14070</td>
      <td>0.41860</td>
      <td>0.65990</td>
      <td>0.25420</td>
      <td>0.2929</td>
      <td>0.09873</td>
    </tr>
    <tr>
      <th>564</th>
      <td>926424</td>
      <td>M</td>
      <td>21.560</td>
      <td>22.39</td>
      <td>142.00</td>
      <td>1479.0</td>
      <td>0.11100</td>
      <td>0.11590</td>
      <td>0.243900</td>
      <td>0.138900</td>
      <td>...</td>
      <td>25.450</td>
      <td>26.40</td>
      <td>166.10</td>
      <td>2027.0</td>
      <td>0.14100</td>
      <td>0.21130</td>
      <td>0.41070</td>
      <td>0.22160</td>
      <td>0.2060</td>
      <td>0.07115</td>
    </tr>
    <tr>
      <th>565</th>
      <td>926682</td>
      <td>M</td>
      <td>20.130</td>
      <td>28.25</td>
      <td>131.20</td>
      <td>1261.0</td>
      <td>0.09780</td>
      <td>0.10340</td>
      <td>0.144000</td>
      <td>0.097910</td>
      <td>...</td>
      <td>23.690</td>
      <td>38.25</td>
      <td>155.00</td>
      <td>1731.0</td>
      <td>0.11660</td>
      <td>0.19220</td>
      <td>0.32150</td>
      <td>0.16280</td>
      <td>0.2572</td>
      <td>0.06637</td>
    </tr>
    <tr>
      <th>566</th>
      <td>926954</td>
      <td>M</td>
      <td>16.600</td>
      <td>28.08</td>
      <td>108.30</td>
      <td>858.1</td>
      <td>0.08455</td>
      <td>0.10230</td>
      <td>0.092510</td>
      <td>0.053020</td>
      <td>...</td>
      <td>18.980</td>
      <td>34.12</td>
      <td>126.70</td>
      <td>1124.0</td>
      <td>0.11390</td>
      <td>0.30940</td>
      <td>0.34030</td>
      <td>0.14180</td>
      <td>0.2218</td>
      <td>0.07820</td>
    </tr>
    <tr>
      <th>567</th>
      <td>927241</td>
      <td>M</td>
      <td>20.600</td>
      <td>29.33</td>
      <td>140.10</td>
      <td>1265.0</td>
      <td>0.11780</td>
      <td>0.27700</td>
      <td>0.351400</td>
      <td>0.152000</td>
      <td>...</td>
      <td>25.740</td>
      <td>39.42</td>
      <td>184.60</td>
      <td>1821.0</td>
      <td>0.16500</td>
      <td>0.86810</td>
      <td>0.93870</td>
      <td>0.26500</td>
      <td>0.4087</td>
      <td>0.12400</td>
    </tr>
    <tr>
      <th>568</th>
      <td>92751</td>
      <td>B</td>
      <td>7.760</td>
      <td>24.54</td>
      <td>47.92</td>
      <td>181.0</td>
      <td>0.05263</td>
      <td>0.04362</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>9.456</td>
      <td>30.37</td>
      <td>59.16</td>
      <td>268.6</td>
      <td>0.08996</td>
      <td>0.06444</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.2871</td>
      <td>0.07039</td>
    </tr>
  </tbody>
</table>
<p>569 rows × 32 columns</p>
</div>



### data pre-processing

1. check missing value

2. select attribute: id 제거

3. box plot 을 통한 data scale(SD, mean) 평가




```python
# null 값 찾기
pd.isnull(table).any()
```




    ID                         False
    diagnosis                  False
    radius                     False
    texture                    False
    perimeter                  False
    area                       False
    smoothness                 False
    compactness                False
    concavity                  False
    concave points             False
    symmetry                   False
    fractal dimention          False
    radius_SD                  False
    texture_SD                 False
    perimeter_SD               False
    area_SD                    False
    smoothness_SD              False
    compactness_SD             False
    concavity_SD               False
    concave points_SD          False
    symmetry_SD                False
    fractal dimention_SD       False
    radius_worst               False
    texture_worst              False
    perimeter_worst            False
    area_worst                 False
    smoothness_worst           False
    compactness_worst          False
    concavity_worst            False
    concave points_worst       False
    symmetry_worst             False
    fractal dimention_worst    False
    dtype: bool




```python
t_a = table.iloc[:,1:]
```


```python
import seaborn as sns
sns.countplot(table["diagnosis"])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x21c6a5b0518>



위 plot을 보면 이 데이터의 클라스 분포를 알수 있다.


```python
t_a.boxplot(column=name[2:12],figsize=(18,5))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x21c6c7b9400>




![png](output_11_1.png)



```python
t_a.boxplot(column=name[12:22],figsize=(18,5))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x21c6cac44e0>




![png](output_12_1.png)



```python
t_a.boxplot(column=name[22:32],figsize=(18,5))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x21c6c971e48>




![png](output_13_1.png)


we can see that area, area_SD, area_worst have dominated scale

in order to see other attribute's scale, i will get rid of area, area_SD, area_worst and make a box_plot


```python
t_not_area = t_a.drop([ 'area', 'area_SD', 'area_worst' ],axis=1)
```


```python
t_not_area.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>diagnosis</th>
      <th>radius</th>
      <th>texture</th>
      <th>perimeter</th>
      <th>smoothness</th>
      <th>compactness</th>
      <th>concavity</th>
      <th>concave points</th>
      <th>symmetry</th>
      <th>fractal dimention</th>
      <th>...</th>
      <th>fractal dimention_SD</th>
      <th>radius_worst</th>
      <th>texture_worst</th>
      <th>perimeter_worst</th>
      <th>smoothness_worst</th>
      <th>compactness_worst</th>
      <th>concavity_worst</th>
      <th>concave points_worst</th>
      <th>symmetry_worst</th>
      <th>fractal dimention_worst</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M</td>
      <td>17.99</td>
      <td>10.38</td>
      <td>122.80</td>
      <td>0.11840</td>
      <td>0.27760</td>
      <td>0.3001</td>
      <td>0.14710</td>
      <td>0.2419</td>
      <td>0.07871</td>
      <td>...</td>
      <td>0.006193</td>
      <td>25.38</td>
      <td>17.33</td>
      <td>184.60</td>
      <td>0.1622</td>
      <td>0.6656</td>
      <td>0.7119</td>
      <td>0.2654</td>
      <td>0.4601</td>
      <td>0.11890</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M</td>
      <td>20.57</td>
      <td>17.77</td>
      <td>132.90</td>
      <td>0.08474</td>
      <td>0.07864</td>
      <td>0.0869</td>
      <td>0.07017</td>
      <td>0.1812</td>
      <td>0.05667</td>
      <td>...</td>
      <td>0.003532</td>
      <td>24.99</td>
      <td>23.41</td>
      <td>158.80</td>
      <td>0.1238</td>
      <td>0.1866</td>
      <td>0.2416</td>
      <td>0.1860</td>
      <td>0.2750</td>
      <td>0.08902</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M</td>
      <td>19.69</td>
      <td>21.25</td>
      <td>130.00</td>
      <td>0.10960</td>
      <td>0.15990</td>
      <td>0.1974</td>
      <td>0.12790</td>
      <td>0.2069</td>
      <td>0.05999</td>
      <td>...</td>
      <td>0.004571</td>
      <td>23.57</td>
      <td>25.53</td>
      <td>152.50</td>
      <td>0.1444</td>
      <td>0.4245</td>
      <td>0.4504</td>
      <td>0.2430</td>
      <td>0.3613</td>
      <td>0.08758</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M</td>
      <td>11.42</td>
      <td>20.38</td>
      <td>77.58</td>
      <td>0.14250</td>
      <td>0.28390</td>
      <td>0.2414</td>
      <td>0.10520</td>
      <td>0.2597</td>
      <td>0.09744</td>
      <td>...</td>
      <td>0.009208</td>
      <td>14.91</td>
      <td>26.50</td>
      <td>98.87</td>
      <td>0.2098</td>
      <td>0.8663</td>
      <td>0.6869</td>
      <td>0.2575</td>
      <td>0.6638</td>
      <td>0.17300</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M</td>
      <td>20.29</td>
      <td>14.34</td>
      <td>135.10</td>
      <td>0.10030</td>
      <td>0.13280</td>
      <td>0.1980</td>
      <td>0.10430</td>
      <td>0.1809</td>
      <td>0.05883</td>
      <td>...</td>
      <td>0.005115</td>
      <td>22.54</td>
      <td>16.67</td>
      <td>152.20</td>
      <td>0.1374</td>
      <td>0.2050</td>
      <td>0.4000</td>
      <td>0.1625</td>
      <td>0.2364</td>
      <td>0.07678</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 28 columns</p>
</div>




```python
t_not_area.iloc[:,1:10].boxplot(figsize=(18,5))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x21c6cfb2400>




![png](output_17_1.png)



```python
t_not_area.iloc[:,10:19].boxplot(figsize=(18,5))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x21c6cd10e80>




![png](output_18_1.png)



```python
t_not_area.iloc[:,19:28].boxplot(figsize=(18,5))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x21c6ce1c780>




![png](output_19_1.png)


### data transformation

##### data transformation to 
1. mean = 0, sd = 1
2. min_max scale [0,1]
3. max_abs scale [-1,1]
4. normalization



```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np

# data set attribute / class로 나누기
x_ori = t_a.iloc[:,1:]
y_ori = t_a.iloc[:,0]

# data test set, train set으로 분리
x_train,x_test,y_train,y_test = train_test_split(x_ori,y_ori,test_size = 0.3,random_state =0 )

# scale function mean=0, sd = 1
scaler = preprocessing.StandardScaler().fit(x_ori)
scale_x_test = scaler.transform(x_test)
scale_x_train = scaler.transform(x_train)
scale_x = scaler.transform(x_ori)


# min_max_scaler  [0,1]
min_max_scaler = preprocessing.MinMaxScaler().fit(x_ori)
min_max_x_test = min_max_scaler.transform(x_test)
min_max_x_train= min_max_scaler.transform(x_train)
min_max_x = min_max_scaler.transform(x_ori)

# max_abs_scaler
max_abs_scaler=preprocessing.MaxAbsScaler().fit(x_ori)
max_abs_x_test = max_abs_scaler.transform(x_test)
max_abs_x_train= max_abs_scaler.transform(x_train)
max_abs_x = max_abs_scaler.transform(x_ori)

# normalization
normalization=preprocessing.Normalizer().fit(x_ori)
normalization_x_test = normalization.transform(x_test)
normalization_x_train= normalization.transform(x_train)
normalization_x = normalization.transform(x_ori)

```

### for feature selection

1. draw a correaltion matrix in order to find dependency among attributes
2. do a PCA with all test sets ( scale, maxmin, maxabs, norm ) and select a number of pca principle component


```python
corr4 = t_a.corr()
sns.heatmap(corr4)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x21c6d98ef98>




![png](output_23_1.png)


위에서 만든 correlation matrix로 직접 feature selection을 하려고 했으니 어느 정도 상관관계 까지 지워야 할지 확실한 기준이 없기 때문에, wrapper apporach를 통해 나중에 비중있게 사용된 attribute을 고려 할 것이다.

이 프로젝트의 목적은 예측력 높은 모델을 생성하는 것이 목적이라고 말했는데, 단순하게 scaled된 데이터를 가지고 하는 것 뿐만 아니라
PCA를 통해 새로운 feature를 생성해내서 그를 가지고 model을 생성할 것이다



```python
# PCA
from sklearn.decomposition import PCA

########## normal #############################
pca = list()
pca_explained_variance_ratio = list()

for i in range(1,31):
    scale = PCA(n_components=i)
    scale.fit_transform(x_ori)
    pca.append(scale)
    pca_explained_variance_ratio.append(scale.explained_variance_ratio_.tolist())
    
    
pca_ratio = pd.DataFrame(pca_explained_variance_ratio)   

toplot_norm = pca_ratio.T.sum()
print(toplot_norm[toplot_norm>0.95].head(1))

toplot_norm.plot(title = "explained_variance_ratio normal ",grid=True)
```

    0    0.982045
    dtype: float64
    




    <matplotlib.axes._subplots.AxesSubplot at 0x21c6dc95e48>




![png](output_25_2.png)



```python
# PCA
from sklearn.decomposition import PCA

########## scale #############################
pca = list()
pca_explained_variance_ratio = list()

for i in range(1,31):
    scale = PCA(n_components=i)
    scale.fit_transform(scale_x)
    pca.append(scale)
    pca_explained_variance_ratio.append(scale.explained_variance_ratio_.tolist())
    
    
pca_ratio = pd.DataFrame(pca_explained_variance_ratio)   

toplot_scale = pca_ratio.T.sum()
print(toplot_scale[toplot_scale>0.95].head(2))
toplot_scale.plot(title = "explained_variance_ratio scaled",grid=True)
```

    9     0.951569
    10    0.961366
    dtype: float64
    




    <matplotlib.axes._subplots.AxesSubplot at 0x21c6dd06a58>




![png](output_26_2.png)



```python

########## MIN MAX ###############

pca = list()
pca_explained_variance_ratio = list()

for i in range(1,31):
    minmax = PCA(n_components=i)
    minmax.fit_transform(min_max_x)
    pca.append(minmax)
    pca_explained_variance_ratio.append(minmax.explained_variance_ratio_.tolist())
pca_ratio = pd.DataFrame(pca_explained_variance_ratio)    

toplot_minmax = pca_ratio.T.sum()
print(toplot_minmax[toplot_minmax>0.95].head(2))
toplot_minmax.plot(title = "explained_variance_ratio minmax",grid=True)

```

    9     0.957706
    10    0.966200
    dtype: float64
    




    <matplotlib.axes._subplots.AxesSubplot at 0x21c6ddab2e8>




![png](output_27_2.png)



```python
pca = list()
pca_explained_variance_ratio = list()

for i in range(1,31):
    maxabs = PCA(n_components=i)
    maxabs.fit_transform(max_abs_x)
    pca.append(maxabs)
    pca_explained_variance_ratio.append(maxabs.explained_variance_ratio_.tolist())
pca_ratio = pd.DataFrame(pca_explained_variance_ratio)    

toplot_maxabs = pca_ratio.T.sum()
print(toplot_maxabs[toplot_maxabs>0.95].head(2))
toplot_maxabs.plot(title = "explained_variance_ratio maxabs",grid=True)
```

    8    0.950385
    9    0.960566
    dtype: float64
    




    <matplotlib.axes._subplots.AxesSubplot at 0x21c6de074e0>




![png](output_28_2.png)



```python
pca = list()
pca_explained_variance_ratio = list()

for i in range(1,31):
    norm = PCA(n_components=i)
    norm.fit_transform(normalization_x)
    pca.append(norm)
    pca_explained_variance_ratio.append(norm.explained_variance_ratio_.tolist())
pca_ratio = pd.DataFrame(pca_explained_variance_ratio)    

toplot_norm = pca_ratio.T.sum()
print(toplot_norm[toplot_norm>0.95].head(2))
toplot_norm.plot(title = "explained_variance_ratio norm ",grid=True)
```

    2    0.984753
    3    0.996886
    dtype: float64
    




    <matplotlib.axes._subplots.AxesSubplot at 0x21c6ee79550>




![png](output_29_2.png)



```python
k_n = 1
k = PCA(n_components=k_n)
k= k.fit_transform(x_ori)
pca_norm= pd.DataFrame(data=k,columns = ["comp"+str(i+1) for i in range(k_n)] )
pca_norm_ = pd.concat([pca_norm,y_ori],axis=1)

k_n = 9
k = PCA(n_components=k_n)
k= k.fit_transform(scale_x)
pca_scale= pd.DataFrame(data=k,columns =["comp"+str(i+1) for i in range(k_n)] )
pca_scale_ = pd.concat([pca_scale,y_ori],axis=1)

k_n = 9
k = PCA(n_components=k_n)
k=k.fit_transform(min_max_x)
pca_minmax= pd.DataFrame(data=k,columns = ["comp"+str(i+1) for i in range(k_n)] )
pca_minmax_ = pd.concat([pca_minmax,y_ori],axis=1)

k_n = 8
k = PCA(n_components=k_n)
k=k.fit_transform(max_abs_x)
pca_maxabs= pd.DataFrame(data=k,columns = ["comp"+str(i+1) for i in range(k_n)] )
pca_maxabs_ = pd.concat([pca_maxabs,y_ori],axis=1)

k_n = 2
k = PCA(n_components=k_n)
k=k.fit_transform(max_abs_x)
pca_normal= pd.DataFrame(data=k,columns = ['comp1','comp2'] )
pca_normal_ = pd.concat([pca_norm,y_ori],axis=1)

```

지금 까지 data preprocessing 으로 mead = 0, sd =1 / min max / max abs / normalization 에 해당하는 데이터 셋을 따로 만들었고, 
feature selection으로 ID attribute를 없엤고,
feature creation의 일종인 PCA를 통해 새로운 table을 만들었다.

이제 이를 이용하여 다양한 classifier 를 만들것이다

decision tree 와 random forest를 우선 사용할 것인데 이 모델에서는 예측력을 높이는 것도 목적이지만, 모델을 해석하여 어떤 attribute가 영양을 많이 끼치는지 알아보기 위해서 PCA는 사용하지 않을 것이다.


```python
# decision tree

from sklearn import tree

decisionTree_entropy = []
decisionTree_gini = []

###################### normal data #################################


# 트리 생성_entropy
diagnosis_tree_entropy = tree.DecisionTreeClassifier(criterion="entropy",random_state = 0)
diagnosis_tree_entropy.fit(x_train,y_train)

# 트리 생성_gini
diagnosis_tree_gini = tree.DecisionTreeClassifier(criterion="gini",random_state = 0)
diagnosis_tree_gini.fit(x_train,y_train)


# accuracy 구하기
from sklearn.metrics import accuracy_score

# entropy
dia_predict = diagnosis_tree_entropy.predict(x_test)
print('Accuracy of "normal" by criterion = entropy: %.4f' % accuracy_score(y_test, dia_predict))
decisionTree_entropy.append(accuracy_score(y_test, dia_predict))

# gini
dia_predict = diagnosis_tree_gini.predict(x_test)
print('Accuracy of "normal" by criterion = gini: %.4f' % accuracy_score(y_test, dia_predict))
decisionTree_gini.append(accuracy_score(y_test, dia_predict))

#############################  mean = 0, sd = 1 data################################

x_train_ = scale_x_train
x_test_ = scale_x_test

# 트리 생성_entropy
diagnosis_tree_entropy1 = tree.DecisionTreeClassifier(criterion="entropy",random_state = 0)
diagnosis_tree_entropy1.fit(x_train_,y_train)

# 트리 생성_gini
diagnosis_tree_gini1 = tree.DecisionTreeClassifier(criterion="gini",random_state = 0)
diagnosis_tree_gini1.fit(x_train_,y_train)


# accuracy 구하기
from sklearn.metrics import accuracy_score

# entropy
dia_predict = diagnosis_tree_entropy1.predict(x_test_)
print('Accuracy of "scale" by criterion = entropy: %.4f' % accuracy_score(y_test, dia_predict))
decisionTree_entropy.append(accuracy_score(y_test, dia_predict))

# gini
dia_predict = diagnosis_tree_gini1.predict(x_test_)
print('Accuracy of "scale" by criterion = gini: %.4f' % accuracy_score(y_test, dia_predict))
decisionTree_gini.append(accuracy_score(y_test, dia_predict))
######################  min max  [0,1]  data #############################

x_train_ = min_max_x_train
x_test_ = min_max_x_test

# 트리 생성_entropy
diagnosis_tree_entropy11 = tree.DecisionTreeClassifier(criterion="entropy",random_state = 0)
diagnosis_tree_entropy11.fit(x_train_,y_train)

# 트리 생성_gini
diagnosis_tree_gini11 = tree.DecisionTreeClassifier(criterion="gini",random_state = 0)
diagnosis_tree_gini11.fit(x_train_,y_train)


# accuracy 구하기
from sklearn.metrics import accuracy_score

# entropy
dia_predict = diagnosis_tree_entropy11.predict(x_test_)
print('Accuracy of "minmax" by criterion = entropy: %.4f' % accuracy_score(y_test, dia_predict))
decisionTree_entropy.append(accuracy_score(y_test, dia_predict))

# gini
dia_predict = diagnosis_tree_gini11.predict(x_test_)
print('Accuracy of "minmax" by criterion = gini: %.4f' % accuracy_score(y_test, dia_predict))
decisionTree_gini.append(accuracy_score(y_test, dia_predict))

#################### max abs [-1,1]  data  #####################################

x_train_ = max_abs_x_train
x_test_ = max_abs_x_test

# 트리 생성_entropy
diagnosis_tree_entropy12 = tree.DecisionTreeClassifier(criterion="entropy",random_state = 0)
diagnosis_tree_entropy12.fit(x_train_,y_train)

# 트리 생성_gini
diagnosis_tree_gini12 = tree.DecisionTreeClassifier(criterion="gini",random_state = 0)
diagnosis_tree_gini12.fit(x_train_,y_train)


# accuracy 구하기
from sklearn.metrics import accuracy_score

# entropy
dia_predict = diagnosis_tree_entropy12.predict(x_test_)
print('Accuracy of "max abs" by criterion = entropy: %.4f' % accuracy_score(y_test, dia_predict))
decisionTree_entropy.append(accuracy_score(y_test, dia_predict))

# gini
dia_predict = diagnosis_tree_gini12.predict(x_test_)
print('Accuracy of "max abs" by criterion = gini: %.4f' % accuracy_score(y_test, dia_predict))
decisionTree_gini.append(accuracy_score(y_test, dia_predict))

####################### normalization data #########################################

x_train_ = normalization_x_train
x_test_ = normalization_x_test

# 트리 생성_entropy
diagnosis_tree_entropy13 = tree.DecisionTreeClassifier(criterion="entropy",random_state = 0)
diagnosis_tree_entropy13.fit(x_train_,y_train)

# 트리 생성_gini
diagnosis_tree_gini13 = tree.DecisionTreeClassifier(criterion="gini",random_state = 0)
diagnosis_tree_gini13.fit(x_train_,y_train)


# accuracy 구하기
from sklearn.metrics import accuracy_score

# entropy
dia_predict = diagnosis_tree_entropy13.predict(x_test_)
print('Accuracy of "normalization" by criterion = entropy: %.4f' % accuracy_score(y_test, dia_predict))
decisionTree_entropy.append(accuracy_score(y_test, dia_predict))

# gini
dia_predict = diagnosis_tree_gini13.predict(x_test_)
print('Accuracy of "normalization" by criterion = gini: %.4f' % accuracy_score(y_test, dia_predict))
decisionTree_gini.append(accuracy_score(y_test, dia_predict))
```

    Accuracy of "normal" by criterion = entropy: 0.9357
    Accuracy of "normal" by criterion = gini: 0.9123
    Accuracy of "scale" by criterion = entropy: 0.9357
    Accuracy of "scale" by criterion = gini: 0.9123
    Accuracy of "minmax" by criterion = entropy: 0.9357
    Accuracy of "minmax" by criterion = gini: 0.9123
    Accuracy of "max abs" by criterion = entropy: 0.9357
    Accuracy of "max abs" by criterion = gini: 0.9123
    Accuracy of "normalization" by criterion = entropy: 0.9357
    Accuracy of "normalization" by criterion = gini: 0.9357
    


```python
# random forest

from sklearn.ensemble import RandomForestClassifier

randomForest_gini = []
randomForest_entropy = []

###################### normal data #################################


# random forest 생성_gini
rf_gini = RandomForestClassifier(random_state = 0).fit(x_train,y_train.values.ravel())
# random forest_entropy
rf_entropy = RandomForestClassifier(criterion="entropy" ,random_state = 0).fit(x_train,y_train.values.ravel())
# accuracy 구하기
g=rf_gini.score(x_test,y_test.values.ravel())
e=rf_entropy.score(x_test,y_test.values.ravel())
print('Accuracy of "normal Random Forest" by criterion = entropy: %.4f' % e)
print('Accuracy of "normal Random Forest" by criterion = gini: %.4f' % g)

randomForest_gini.append(g)
randomForest_entropy.append(e)
#############################  mean = 0, sd = 1 data################################

x_train_ = scale_x_train
x_test_ = scale_x_test


# random forest 생성_gini
rf_gini = RandomForestClassifier(random_state = 0).fit(x_train_,y_train.values.ravel())
# random forest_entropy
rf_entropy = RandomForestClassifier(criterion="entropy" ,random_state = 0).fit(x_train_,y_train.values.ravel())
# accuracy 구하기
g=rf_gini.score(x_test_,y_test.values.ravel())
e=rf_entropy.score(x_test_,y_test.values.ravel())
print('Accuracy of "scaled Random Forest" by criterion = entropy: %.4f' % e)
print('Accuracy of "scaled Random Forest" by criterion = gini: %.4f' % g)

randomForest_gini.append(g)
randomForest_entropy.append(e)
######################  min max  [0,1]  data #############################

x_train_ = min_max_x_train
x_test_ = min_max_x_test

# random forest 생성_gini
rf_gini = RandomForestClassifier(random_state = 0).fit(x_train_,y_train.values.ravel())
# random forest_entropy
rf_entropy = RandomForestClassifier(criterion="entropy" ,random_state = 0).fit(x_train_,y_train.values.ravel())
# accuracy 구하기
g=rf_gini.score(x_test_,y_test.values.ravel())
e=rf_entropy.score(x_test_,y_test.values.ravel())
print('Accuracy of "min max Random Forest" by criterion = entropy: %.4f' % e)
print('Accuracy of "min max Random Forest" by criterion = gini: %.4f' % g)

randomForest_gini.append(g)
randomForest_entropy.append(e)
#################### max abs [-1,1]  data  #####################################

x_train_ = max_abs_x_train
x_test_ = max_abs_x_test

# random forest 생성_gini
rf_gini = RandomForestClassifier(random_state = 0).fit(x_train_,y_train.values.ravel())
# random forest_entropy
rf_entropy = RandomForestClassifier(criterion="entropy" ,random_state = 0).fit(x_train_,y_train.values.ravel())
# accuracy 구하기
g=rf_gini.score(x_test_,y_test.values.ravel())
e=rf_entropy.score(x_test_,y_test.values.ravel())
print('Accuracy of "max abs Random Forest" by criterion = entropy: %.4f' % e)
print('Accuracy of "max abs Random Forest" by criterion = gini: %.4f' % g)

randomForest_gini.append(g)
randomForest_entropy.append(e)
####################### normalization data #########################################

x_train_ = normalization_x_train
x_test_ = normalization_x_test

# random forest 생성_gini
rf_gini = RandomForestClassifier(random_state = 0).fit(x_train_,y_train.values.ravel())
# random forest_entropy
rf_entropy = RandomForestClassifier(criterion="entropy" ,random_state = 0).fit(x_train_,y_train.values.ravel())
# accuracy 구하기
g=rf_gini.score(x_test_,y_test.values.ravel())
e=rf_entropy.score(x_test_,y_test.values.ravel())
print('Accuracy of "normalization Random Forest" by criterion = entropy: %.4f' % e)
print('Accuracy of "normalization Random Forest" by criterion = gini: %.4f' % g)

randomForest_gini.append(g)
randomForest_entropy.append(e)

```

    C:\Users\Lee Joo Ye\Anaconda3\lib\site-packages\sklearn\ensemble\weight_boosting.py:29: DeprecationWarning: numpy.core.umath_tests is an internal NumPy module and should not be imported. It will be removed in a future NumPy release.
      from numpy.core.umath_tests import inner1d
    

    Accuracy of "normal Random Forest" by criterion = entropy: 0.9649
    Accuracy of "normal Random Forest" by criterion = gini: 0.9649
    Accuracy of "scaled Random Forest" by criterion = entropy: 0.9649
    Accuracy of "scaled Random Forest" by criterion = gini: 0.9649
    Accuracy of "min max Random Forest" by criterion = entropy: 0.9649
    Accuracy of "min max Random Forest" by criterion = gini: 0.9649
    Accuracy of "max abs Random Forest" by criterion = entropy: 0.9649
    Accuracy of "max abs Random Forest" by criterion = gini: 0.9649
    Accuracy of "normalization Random Forest" by criterion = entropy: 0.9415
    Accuracy of "normalization Random Forest" by criterion = gini: 0.9591
    


```python
decisionTree_acc = pd.DataFrame()

decisionTree_acc["gini"] =decisionTree_gini
decisionTree_acc["entropy"] = decisionTree_entropy 

decisionTree_acc.index = ["ori","scale","minmax","max abs","normalize"]

randomForest_acc = pd.DataFrame()

randomForest_acc["gini"] = randomForest_gini
randomForest_acc["entropy"]=randomForest_entropy
randomForest_acc.index = ["ori","scale","minmax","max abs","normalize"]
```


```python
decisionTree_acc
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gini</th>
      <th>entropy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ori</th>
      <td>0.912281</td>
      <td>0.935673</td>
    </tr>
    <tr>
      <th>scale</th>
      <td>0.912281</td>
      <td>0.935673</td>
    </tr>
    <tr>
      <th>minmax</th>
      <td>0.912281</td>
      <td>0.935673</td>
    </tr>
    <tr>
      <th>max abs</th>
      <td>0.912281</td>
      <td>0.935673</td>
    </tr>
    <tr>
      <th>normalize</th>
      <td>0.935673</td>
      <td>0.935673</td>
    </tr>
  </tbody>
</table>
</div>




```python
randomForest_acc
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gini</th>
      <th>entropy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ori</th>
      <td>0.964912</td>
      <td>0.964912</td>
    </tr>
    <tr>
      <th>scale</th>
      <td>0.964912</td>
      <td>0.964912</td>
    </tr>
    <tr>
      <th>minmax</th>
      <td>0.964912</td>
      <td>0.964912</td>
    </tr>
    <tr>
      <th>max abs</th>
      <td>0.964912</td>
      <td>0.964912</td>
    </tr>
    <tr>
      <th>normalize</th>
      <td>0.959064</td>
      <td>0.941520</td>
    </tr>
  </tbody>
</table>
</div>



k-nearest neighbors classifier는 PCA를 활용하여 모델을 추가로 생성했다.


```python

k1_ac = []

from sklearn.neighbors import KNeighborsClassifier
k = 1
####################################### normal ###################################
knn1 = KNeighborsClassifier(n_neighbors = k,weights='distance')
knn1.fit(x_train,y_train)
print("accuracy of 1-nearest neighbors by normal data : ",knn1.score(x_test,y_test))

k1_ac.append(knn1.score(x_test,y_test))

#############################  mean = 0, sd = 1 data################################

x_train_ = scale_x_train
x_test_ = scale_x_test
knn1 = KNeighborsClassifier(n_neighbors = k,weights='distance')
knn1.fit(x_train_,y_train)
print("accuracy of 1-nearest neighbors by scaled data: ",knn1.score(x_test_,y_test))

k1_ac.append(knn1.score(x_test_,y_test))

######################  min max  [0,1]  data #############################

x_train_ = min_max_x_train
x_test_ = min_max_x_test
knn1 = KNeighborsClassifier(n_neighbors = k,weights='distance')
knn1.fit(x_train_,y_train)
print("accuracy of 1-nearest neighbors by min max data: ",knn1.score(x_test_,y_test))

k1_ac.append(knn1.score(x_test_,y_test))

#################### max abs [-1,1]  data  #####################################

x_train_ = max_abs_x_train
x_test_ = max_abs_x_test
knn1 = KNeighborsClassifier(n_neighbors = k,weights='distance')
knn1.fit(x_train_,y_train)
print("accuracy of 1-nearest neighbors by max abs  data: ",knn1.score(x_test_,y_test))

k1_ac.append(knn1.score(x_test_,y_test))

####################### normalization data #########################################

x_train_ = normalization_x_train
x_test_ = normalization_x_test
knn1 = KNeighborsClassifier(n_neighbors = k,weights='distance')
knn1.fit(x_train_,y_train)
print("accuracy of 1-nearest neighbors : by normalization ",knn1.score(x_test_,y_test))

k1_ac.append(knn1.score(x_test_,y_test))

```

    accuracy of 1-nearest neighbors by normal data :  0.9181286549707602
    accuracy of 1-nearest neighbors by scaled data:  0.935672514619883
    accuracy of 1-nearest neighbors by min max data:  0.9298245614035088
    accuracy of 1-nearest neighbors by max abs  data:  0.9473684210526315
    accuracy of 1-nearest neighbors : by normalization  0.8596491228070176
    


```python
k1_ac
```




    [0.9181286549707602,
     0.935672514619883,
     0.9298245614035088,
     0.9473684210526315,
     0.8596491228070176]




```python
k=1
######################### PCA ###############################
k1_pca_ac = []

x_train_, x_test_,y_train_,y_test_ = train_test_split(pca_norm,y_ori,test_size = 0.3,random_state =0 )
knn1 = KNeighborsClassifier(n_neighbors = k,weights='distance')
knn1.fit(x_train_,y_train)
print("accuracy of 1-nearest neighbors : by PCA 1 normal ",knn1.score(x_test_,y_test_))
k1_pca_ac.append(knn1.score(x_test_,y_test_))


x_train_, x_test_,y_train_,y_test_ = train_test_split(pca_scale,y_ori,test_size = 0.3,random_state =0 )
knn1 = KNeighborsClassifier(n_neighbors = k,weights='distance')
knn1.fit(x_train_,y_train)
print("accuracy of 1-nearest neighbors : by PCA 13 scaled",knn1.score(x_test_,y_test_))
k1_pca_ac.append(knn1.score(x_test_,y_test_))

x_train_, x_test_,y_train_,y_test_ = train_test_split(pca_minmax,y_ori,test_size = 0.3,random_state =0 )
knn1 = KNeighborsClassifier(n_neighbors = k,weights='distance')
knn1.fit(x_train_,y_train)
print("accuracy of 1-nearest neighbors : by PCA min max 12 ",knn1.score(x_test_,y_test_))
k1_pca_ac.append(knn1.score(x_test_,y_test_))


x_train_, x_test_,y_train_,y_test_ = train_test_split(pca_maxabs,y_ori,test_size = 0.3,random_state =0 )
knn1 = KNeighborsClassifier(n_neighbors = k,weights='distance')
knn1.fit(x_train_,y_train)
print("accuracy of 1-nearest neighbors : by PCA max abs 12 ",knn1.score(x_test_,y_test_))
k1_pca_ac.append(knn1.score(x_test_,y_test_))

x_train_, x_test_ ,y_train_,y_test_= train_test_split(pca_normal,y_ori,test_size = 0.3,random_state =0 )
knn1 = KNeighborsClassifier(n_neighbors = k,weights='distance')
knn1.fit(x_train_,y_train)
print("accuracy of 1-nearest neighbors : by PCA 2  normalization",knn1.score(x_test_,y_test_))
k1_pca_ac.append(knn1.score(x_test_,y_test_))
```

    accuracy of 1-nearest neighbors : by PCA 1 normal  0.8362573099415205
    accuracy of 1-nearest neighbors : by PCA 13 scaled 0.9415204678362573
    accuracy of 1-nearest neighbors : by PCA min max 12  0.9415204678362573
    accuracy of 1-nearest neighbors : by PCA max abs 12  0.9473684210526315
    accuracy of 1-nearest neighbors : by PCA 2  normalization 0.8888888888888888
    


```python
k = 2
k2_ac = []

####################################### normal ###################################
knn2 = KNeighborsClassifier(n_neighbors = k,weights='distance')
knn2.fit(x_train,y_train)
print("accuracy of 2-nearest neighbors by normal data : ",knn2.score(x_test,y_test))
k2_ac.append(knn2.score(x_test,y_test))

#############################  mean = 0, sd = 1 data################################

x_train_ = scale_x_train
x_test_ = scale_x_test
knn2 = KNeighborsClassifier(n_neighbors = k,weights='distance')
knn2.fit(x_train_,y_train)
print("accuracy of 2-nearest neighbors by scaled data: ",knn2.score(x_test_,y_test))
k2_ac.append(knn2.score(x_test_,y_test))

######################  min max  [0,1]  data #############################

x_train_ = min_max_x_train
x_test_ = min_max_x_test
knn2 = KNeighborsClassifier(n_neighbors = k,weights='distance')
knn2.fit(x_train_,y_train)
print("accuracy of 2-nearest neighbors by min max data: ",knn2.score(x_test_,y_test))
k2_ac.append(knn2.score(x_test_,y_test))

#################### max abs [-1,1]  data  #####################################

x_train_ = max_abs_x_train
x_test_ = max_abs_x_test
knn2 = KNeighborsClassifier(n_neighbors = k,weights='distance')
knn2.fit(x_train_,y_train)
print("accuracy of 2-nearest neighbors by max abs  data: ",knn2.score(x_test_,y_test))
k2_ac.append(knn2.score(x_test_,y_test))
####################### normalization data #########################################

x_train_ = normalization_x_train
x_test_ = normalization_x_test
knn2 = KNeighborsClassifier(n_neighbors = k,weights='distance')
knn2.fit(x_train_,y_train)
print("accuracy of 2-nearest neighbors : by normalization ",knn2.score(x_test_,y_test))
k2_ac.append(knn2.score(x_test_,y_test))
```

    accuracy of 2-nearest neighbors by normal data :  0.9181286549707602
    accuracy of 2-nearest neighbors by scaled data:  0.935672514619883
    accuracy of 2-nearest neighbors by min max data:  0.9298245614035088
    accuracy of 2-nearest neighbors by max abs  data:  0.9473684210526315
    accuracy of 2-nearest neighbors : by normalization  0.8596491228070176
    


```python
k=2
k2_pca_ac = []
######################### PCA ###############################

x_train_, x_test_,y_train_,y_test_ = train_test_split(pca_norm,y_ori,test_size = 0.3,random_state =0 )
knn2 = KNeighborsClassifier(n_neighbors = k,weights='distance')
knn2.fit(x_train_,y_train)
print("accuracy of 2-nearest neighbors : by PCA 1 normal ",knn2.score(x_test_,y_test_))
k2_pca_ac.append(knn2.score(x_test_,y_test_))

x_train_, x_test_,y_train_,y_test_ = train_test_split(pca_scale,y_ori,test_size = 0.3,random_state =0 )
knn2 = KNeighborsClassifier(n_neighbors = k,weights='distance')
knn2.fit(x_train_,y_train)
print("accuracy of 2-nearest neighbors : by PCA 9 scaled",knn2.score(x_test_,y_test_))
k2_pca_ac.append(knn2.score(x_test_,y_test_))

x_train_, x_test_,y_train_,y_test_ = train_test_split(pca_minmax,y_ori,test_size = 0.3,random_state =0 )
knn2 = KNeighborsClassifier(n_neighbors = k,weights='distance')
knn2.fit(x_train_,y_train)
print("accuracy of 2-nearest neighbors : by PCA min max 9 ",knn2.score(x_test_,y_test_))
k2_pca_ac.append(knn2.score(x_test_,y_test_))

x_train_, x_test_,y_train_,y_test_ = train_test_split(pca_maxabs,y_ori,test_size = 0.3,random_state =0 )
knn2 = KNeighborsClassifier(n_neighbors = k,weights='distance')
knn2.fit(x_train_,y_train)
print("accuracy of 2-nearest neighbors : by PCA max abs 8 ",knn2.score(x_test_,y_test_))
k2_pca_ac.append(knn2.score(x_test_,y_test_))

x_train_, x_test_ ,y_train_,y_test_= train_test_split(pca_normal,y_ori,test_size = 0.3,random_state =0 )
knn2 = KNeighborsClassifier(n_neighbors = k,weights='distance')
knn2.fit(x_train_,y_train)
print("accuracy of 2-nearest neighbors : by PCA 2  normalization",knn2.score(x_test_,y_test_))
k2_pca_ac.append(knn2.score(x_test_,y_test_))
```

    accuracy of 2-nearest neighbors : by PCA 1 normal  0.8362573099415205
    accuracy of 2-nearest neighbors : by PCA 9 scaled 0.9415204678362573
    accuracy of 2-nearest neighbors : by PCA min max 9  0.9415204678362573
    accuracy of 2-nearest neighbors : by PCA max abs 8  0.9473684210526315
    accuracy of 2-nearest neighbors : by PCA 2  normalization 0.8888888888888888
    


```python
k =3
k3_ac = []

####################################### normal ###################################
knn3 = KNeighborsClassifier(n_neighbors = k,weights='distance')
knn3.fit(x_train,y_train)
print("accuracy of 3-nearest neighbors by normal data : ",knn3.score(x_test,y_test))
a = knn3.score(x_test,y_test)
k3_ac.append(a)

#############################  mean = 0, sd = 1 data################################

x_train_ = scale_x_train
x_test_ = scale_x_test
knn3 = KNeighborsClassifier(n_neighbors = k,weights='distance')
knn3.fit(x_train_,y_train)
print("accuracy of 3-nearest neighbors by scaled data: ",knn3.score(x_test_,y_test))
a = knn3.score(x_test_,y_test)
k3_ac.append(a)

######################  min max  [0,1]  data #############################

x_train_ = min_max_x_train
x_test_ = min_max_x_test
knn3 = KNeighborsClassifier(n_neighbors = k,weights='distance')
knn3.fit(x_train_,y_train)
print("accuracy of 3-nearest neighbors by min max data: ",knn3.score(x_test_,y_test))
a = knn3.score(x_test_,y_test)
k3_ac.append(a)

#################### max abs [-1,1]  data  #####################################

x_train_ = max_abs_x_train
x_test_ = max_abs_x_test
knn3 = KNeighborsClassifier(n_neighbors = k,weights='distance')
knn3.fit(x_train_,y_train)
print("accuracy of 3-nearest neighbors by max abs  data: ",knn3.score(x_test_,y_test))
a = knn3.score(x_test_,y_test)
k3_ac.append(a)

####################### normalization data #########################################

x_train_ = normalization_x_train
x_test_ = normalization_x_test
knn3 = KNeighborsClassifier(n_neighbors = k,weights='distance')
knn3.fit(x_train_,y_train)
print("accuracy of 3-nearest neighbors : by normalization ",knn3.score(x_test_,y_test))
a = knn3.score(x_test_,y_test)
k3_ac.append(a)

```

    accuracy of 3-nearest neighbors by normal data :  0.9239766081871345
    accuracy of 3-nearest neighbors by scaled data:  0.9473684210526315
    accuracy of 3-nearest neighbors by min max data:  0.9590643274853801
    accuracy of 3-nearest neighbors by max abs  data:  0.9415204678362573
    accuracy of 3-nearest neighbors : by normalization  0.8830409356725146
    


```python
k=3
k3_pca_ac = []

######################### PCA ###############################

x_train_, x_test_,y_train_,y_test_ = train_test_split(pca_norm,y_ori,test_size = 0.3,random_state =0 )

knn3 = KNeighborsClassifier(n_neighbors = k,weights='distance')
knn3.fit(x_train_,y_train)
print("accuracy of 3-nearest neighbors : by PCA 1 normal ",knn3.score(x_test_,y_test_))
pc = knn3.score(x_test_,y_test_)
k3_pca_ac.append(pc)

x_train_, x_test_,y_train_,y_test_ = train_test_split(pca_scale,y_ori,test_size = 0.3,random_state =0 )

knn3 = KNeighborsClassifier(n_neighbors = k,weights='distance')
knn3.fit(x_train_,y_train)
print("accuracy of 3-nearest neighbors : by PCA 9 scaled",knn3.score(x_test_,y_test_))
pc = knn3.score(x_test_,y_test_)
k3_pca_ac.append(pc)

x_train_, x_test_,y_train_,y_test_ = train_test_split(pca_minmax,y_ori,test_size = 0.3,random_state =0 )

knn3 = KNeighborsClassifier(n_neighbors = k,weights='distance')
knn3.fit(x_train_,y_train)
print("accuracy of 2-nearest neighbors : by PCA min max 9 ",knn3.score(x_test_,y_test_))
pc = knn3.score(x_test_,y_test_)
k3_pca_ac.append(pc)

x_train_, x_test_,y_train_,y_test_ = train_test_split(pca_maxabs,y_ori,test_size = 0.3,random_state =0 )

knn3 = KNeighborsClassifier(n_neighbors = k,weights='distance')
knn3.fit(x_train_,y_train)
print("accuracy of 2-nearest neighbors : by PCA max abs 8 ",knn3.score(x_test_,y_test_))
pc = knn3.score(x_test_,y_test_)
k3_pca_ac.append(pc)

x_train_, x_test_ ,y_train_,y_test_= train_test_split(pca_normal,y_ori,test_size = 0.3,random_state =0 )

knn3 = KNeighborsClassifier(n_neighbors = k,weights='distance')
knn3.fit(x_train_,y_train)
print("accuracy of 3-nearest neighbors : by PCA 2  normalization",knn3.score(x_test_,y_test_))
pc = knn3.score(x_test_,y_test_)
k3_pca_ac.append(pc)
```

    accuracy of 3-nearest neighbors : by PCA 1 normal  0.8888888888888888
    accuracy of 3-nearest neighbors : by PCA 9 scaled 0.9590643274853801
    accuracy of 2-nearest neighbors : by PCA min max 9  0.9590643274853801
    accuracy of 2-nearest neighbors : by PCA max abs 8  0.9473684210526315
    accuracy of 3-nearest neighbors : by PCA 2  normalization 0.8947368421052632
    


```python
k =4
k4_ac = []
####################################### normal ###################################
knn4 = KNeighborsClassifier(n_neighbors = k,weights='distance')
knn4.fit(x_train,y_train)
print("accuracy of 4-nearest neighbors by normal data : ",knn4.score(x_test,y_test))
a = knn4.score(x_test,y_test)
k4_ac.append(a)

#############################  mean = 0, sd = 1 data################################

x_train_ = scale_x_train
x_test_ = scale_x_test
knn4 = KNeighborsClassifier(n_neighbors = k,weights='distance')
knn4.fit(x_train_,y_train)
print("accuracy of 4-nearest neighbors by scaled data: ",knn4.score(x_test_,y_test))
a = knn4.score(x_test_,y_test)
k4_ac.append(a)

######################  min max  [0,1]  data #############################

x_train_ = min_max_x_train
x_test_ = min_max_x_test
knn4 = KNeighborsClassifier(n_neighbors = k,weights='distance')
knn4.fit(x_train_,y_train)
print("accuracy of 4-nearest neighbors by min max data: ",knn4.score(x_test_,y_test))
a = knn4.score(x_test_,y_test)
k4_ac.append(a)

#################### max abs [-1,1]  data  #####################################

x_train_ = max_abs_x_train
x_test_ = max_abs_x_test
knn4 = KNeighborsClassifier(n_neighbors = k,weights='distance')
knn4.fit(x_train_,y_train)
print("accuracy of 4-nearest neighbors by max abs  data: ",knn4.score(x_test_,y_test))
a = knn4.score(x_test_,y_test)
k4_ac.append(a)

####################### normalization data #########################################

x_train_ = normalization_x_train
x_test_ = normalization_x_test
knn4 = KNeighborsClassifier(n_neighbors = k,weights='distance')
knn4.fit(x_train_,y_train)
print("accuracy of 4-nearest neighbors : by normalization ",knn4.score(x_test_,y_test))
a = knn4.score(x_test_,y_test)
k4_ac.append(a)
```

    accuracy of 4-nearest neighbors by normal data :  0.9239766081871345
    accuracy of 4-nearest neighbors by scaled data:  0.9532163742690059
    accuracy of 4-nearest neighbors by min max data:  0.9590643274853801
    accuracy of 4-nearest neighbors by max abs  data:  0.9473684210526315
    accuracy of 4-nearest neighbors : by normalization  0.8888888888888888
    


```python
k=4
k4_pca_ac = []
######################### PCA ###############################

x_train_, x_test_,y_train_,y_test_ = train_test_split(pca_norm,y_ori,test_size = 0.3,random_state =0 )

knn4 = KNeighborsClassifier(n_neighbors = k,weights='distance')
knn4.fit(x_train_,y_train)
print("accuracy of 4-nearest neighbors : by PCA 1 normal ",knn4.score(x_test_,y_test_))
pc = knn4.score(x_test_,y_test_)
k4_pca_ac.append(pc)

x_train_, x_test_,y_train_,y_test_ = train_test_split(pca_scale,y_ori,test_size = 0.3,random_state =0 )

knn4 = KNeighborsClassifier(n_neighbors = k,weights='distance')
knn4.fit(x_train_,y_train)
print("accuracy of 4-nearest neighbors : by PCA 9 scaled",knn4.score(x_test_,y_test_))
pc = knn4.score(x_test_,y_test_)
k4_pca_ac.append(pc)

x_train_, x_test_,y_train_,y_test_ = train_test_split(pca_minmax,y_ori,test_size = 0.3,random_state =0 )

knn4 = KNeighborsClassifier(n_neighbors = k,weights='distance')
knn4.fit(x_train_,y_train)
print("accuracy of 4-nearest neighbors : by PCA min max 9 ",knn4.score(x_test_,y_test_))
pc = knn4.score(x_test_,y_test_)
k4_pca_ac.append(pc)

x_train_, x_test_,y_train_,y_test_ = train_test_split(pca_maxabs,y_ori,test_size = 0.3,random_state =0 )

knn4 = KNeighborsClassifier(n_neighbors = k,weights='distance')
knn4.fit(x_train_,y_train)
print("accuracy of 4-nearest neighbors : by PCA max abs 8 ",knn4.score(x_test_,y_test_))
pc = knn4.score(x_test_,y_test_)
k4_pca_ac.append(pc)

x_train_, x_test_ ,y_train_,y_test_= train_test_split(pca_normal,y_ori,test_size = 0.3,random_state =0 )

knn4 = KNeighborsClassifier(n_neighbors = k,weights='distance')
knn4.fit(x_train_,y_train)
print("accuracy of 4-nearest neighbors : by PCA 2  normalization",knn4.score(x_test_,y_test_))
pc = knn4.score(x_test_,y_test_)
k4_pca_ac.append(pc)
```

    accuracy of 4-nearest neighbors : by PCA 1 normal  0.8947368421052632
    accuracy of 4-nearest neighbors : by PCA 9 scaled 0.9532163742690059
    accuracy of 4-nearest neighbors : by PCA min max 9  0.9590643274853801
    accuracy of 4-nearest neighbors : by PCA max abs 8  0.9532163742690059
    accuracy of 4-nearest neighbors : by PCA 2  normalization 0.8947368421052632
    


```python
k =5
k5_ac = []
####################################### normal ###################################
knn5 = KNeighborsClassifier(n_neighbors = k,weights='distance')
knn5.fit(x_train,y_train)
print("accuracy of 5-nearest neighbors by normal data : ",knn5.score(x_test,y_test))
a = knn5.score(x_test,y_test)
k5_ac.append(a)

#############################  mean = 0, sd = 1 data################################

x_train_ = scale_x_train
x_test_ = scale_x_test
knn5 = KNeighborsClassifier(n_neighbors = k,weights='distance')
knn5.fit(x_train_,y_train)
print("accuracy of 5-nearest neighbors by scaled data: ",knn5.score(x_test_,y_test))
a = knn5.score(x_test_,y_test)
k5_ac.append(a)

######################  min max  [0,1]  data #############################

x_train_ = min_max_x_train
x_test_ = min_max_x_test
knn5 = KNeighborsClassifier(n_neighbors = k,weights='distance')
knn5.fit(x_train_,y_train)
print("accuracy of 5-nearest neighbors by min max data: ",knn5.score(x_test_,y_test))
a = knn5.score(x_test_,y_test)
k5_ac.append(a)

#################### max abs [-1,1]  data  #####################################

x_train_ = max_abs_x_train
x_test_ = max_abs_x_test
knn5 = KNeighborsClassifier(n_neighbors = k,weights='distance')
knn5.fit(x_train_,y_train)
print("accuracy of 5-nearest neighbors by max abs  data: ",knn5.score(x_test_,y_test))
a = knn5.score(x_test_,y_test)
k5_ac.append(a)

####################### normalization data #########################################

x_train_ = normalization_x_train
x_test_ = normalization_x_test
knn5 = KNeighborsClassifier(n_neighbors = k,weights='distance')
knn5.fit(x_train_,y_train)
print("accuracy of 5-nearest neighbors : by normalization ",knn5.score(x_test_,y_test))
a = knn5.score(x_test_,y_test)
k5_ac.append(a)


```

    accuracy of 5-nearest neighbors by normal data :  0.9473684210526315
    accuracy of 5-nearest neighbors by scaled data:  0.9590643274853801
    accuracy of 5-nearest neighbors by min max data:  0.9707602339181286
    accuracy of 5-nearest neighbors by max abs  data:  0.9532163742690059
    accuracy of 5-nearest neighbors : by normalization  0.9239766081871345
    


```python
k=5
k5_pca_ac = []
######################### PCA ###############################

x_train_, x_test_,y_train_,y_test_ = train_test_split(pca_norm,y_ori,test_size = 0.3,random_state =0 )

knn5 = KNeighborsClassifier(n_neighbors = k,weights='distance')
knn5.fit(x_train_,y_train)
print("accuracy of 5-nearest neighbors : by PCA 1 normal ",knn5.score(x_test_,y_test_))
pc = knn5.score(x_test_,y_test_)
k5_pca_ac.append(pc)


x_train_, x_test_,y_train_,y_test_ = train_test_split(pca_scale,y_ori,test_size = 0.3,random_state =0 )

knn5 = KNeighborsClassifier(n_neighbors = k,weights='distance')
knn5.fit(x_train_,y_train)
print("accuracy of 5-nearest neighbors : by PCA 9 scaled",knn5.score(x_test_,y_test_))
pc = knn5.score(x_test_,y_test_)
k5_pca_ac.append(pc)


x_train_, x_test_,y_train_,y_test_ = train_test_split(pca_minmax,y_ori,test_size = 0.3,random_state =0 )

knn5 = KNeighborsClassifier(n_neighbors = k,weights='distance')
knn5.fit(x_train_,y_train)
print("accuracy of 5-nearest neighbors : by PCA min max 9 ",knn5.score(x_test_,y_test_))
pc = knn5.score(x_test_,y_test_)
k5_pca_ac.append(pc)

x_train_, x_test_,y_train_,y_test_ = train_test_split(pca_maxabs,y_ori,test_size = 0.3,random_state =0 )

knn5 = KNeighborsClassifier(n_neighbors = k,weights='distance')
knn5.fit(x_train_,y_train)
print("accuracy of 5-nearest neighbors : by PCA max abs 8 ",knn5.score(x_test_,y_test_))
pc = knn5.score(x_test_,y_test_)
k5_pca_ac.append(pc)

x_train_, x_test_ ,y_train_,y_test_= train_test_split(pca_normal,y_ori,test_size = 0.3,random_state =0 )

knn5 = KNeighborsClassifier(n_neighbors = k,weights='distance')
knn5.fit(x_train_,y_train)
print("accuracy of 5-nearest neighbors : by PCA 2  normalization",knn5.score(x_test_,y_test_))
pc = knn5.score(x_test_,y_test_)
k5_pca_ac.append(pc)

```

    accuracy of 5-nearest neighbors : by PCA 1 normal  0.9064327485380117
    accuracy of 5-nearest neighbors : by PCA 9 scaled 0.9590643274853801
    accuracy of 5-nearest neighbors : by PCA min max 9  0.9649122807017544
    accuracy of 5-nearest neighbors : by PCA max abs 8  0.9532163742690059
    accuracy of 5-nearest neighbors : by PCA 2  normalization 0.9064327485380117
    


```python
k =6
k6_ac = []
####################################### normal ###################################
knn6 = KNeighborsClassifier(n_neighbors = k,weights='distance')
knn6.fit(x_train,y_train)
print("accuracy of 6-nearest neighbors by normal data : ",knn6.score(x_test,y_test))
a = knn6.score(x_test,y_test)
k6_ac.append(a)


#############################  mean = 0, sd = 1 data################################

x_train_ = scale_x_train
x_test_ = scale_x_test
knn6 = KNeighborsClassifier(n_neighbors = k,weights='distance')
knn6.fit(x_train_,y_train)
print("accuracy of 6-nearest neighbors by scaled data: ",knn6.score(x_test_,y_test))
a = knn6.score(x_test_,y_test)
k6_ac.append(a)

######################  min max  [0,1]  data #############################

x_train_ = min_max_x_train
x_test_ = min_max_x_test
knn6 = KNeighborsClassifier(n_neighbors = k,weights='distance')
knn6.fit(x_train_,y_train)
print("accuracy of 6-nearest neighbors by min max data: ",knn6.score(x_test_,y_test))
a = knn6.score(x_test_,y_test)
k6_ac.append(a)

#################### max abs [-1,1]  data  #####################################

x_train_ = max_abs_x_train
x_test_ = max_abs_x_test
knn6 = KNeighborsClassifier(n_neighbors = k,weights='distance')
knn6.fit(x_train_,y_train)
print("accuracy of 6-nearest neighbors by max abs  data: ",knn6.score(x_test_,y_test))
a = knn6.score(x_test_,y_test)
k6_ac.append(a)

####################### normalization data #########################################

x_train_ = normalization_x_train
x_test_ = normalization_x_test
knn6 = KNeighborsClassifier(n_neighbors = k,weights='distance')
knn6.fit(x_train_,y_train)
print("accuracy of 6-nearest neighbors : by normalization ",knn6.score(x_test_,y_test))
a = knn6.score(x_test_,y_test)
k6_ac.append(a)

```

    accuracy of 6-nearest neighbors by normal data :  0.9415204678362573
    accuracy of 6-nearest neighbors by scaled data:  0.9649122807017544
    accuracy of 6-nearest neighbors by min max data:  0.9707602339181286
    accuracy of 6-nearest neighbors by max abs  data:  0.9532163742690059
    accuracy of 6-nearest neighbors : by normalization  0.9298245614035088
    


```python
k=6
k6_pca_ac =[]
######################### PCA ###############################

x_train_, x_test_,y_train_,y_test_ = train_test_split(pca_norm,y_ori,test_size = 0.3,random_state =0 )

knn6 = KNeighborsClassifier(n_neighbors = k,weights='distance')
knn6.fit(x_train_,y_train_)
print("accuracy of 6-nearest neighbors : by PCA 1 normal ",knn6.score(x_test_,y_test_))
pc = knn6.score(x_test_,y_test_)
k6_pca_ac.append(pc)

x_train_, x_test_,y_train_,y_test_ = train_test_split(pca_scale,y_ori,test_size = 0.3,random_state =0 )

knn6 = KNeighborsClassifier(n_neighbors = k,weights='distance')
knn6.fit(x_train_,y_train_)
print("accuracy of 6-nearest neighbors : by PCA 9 scaled",knn6.score(x_test_,y_test_))
pc = knn6.score(x_test_,y_test_)
k6_pca_ac.append(pc)

x_train_, x_test_,y_train_,y_test_ = train_test_split(pca_minmax,y_ori,test_size = 0.3,random_state =0 )

knn6 = KNeighborsClassifier(n_neighbors = k,weights='distance')
knn6.fit(x_train_,y_train_)
print("accuracy of 6-nearest neighbors : by PCA min max 9 ",knn6.score(x_test_,y_test_))
pc = knn6.score(x_test_,y_test_)
k6_pca_ac.append(pc)

x_train_, x_test_,y_train_,y_test_ = train_test_split(pca_maxabs,y_ori,test_size = 0.3,random_state =0 )

knn6 = KNeighborsClassifier(n_neighbors = k,weights='distance')
knn6.fit(x_train_,y_train_)
print("accuracy of 6-nearest neighbors : by PCA max abs 8 ",knn6.score(x_test_,y_test_))
pc = knn6.score(x_test_,y_test_)
k6_pca_ac.append(pc)

x_train_, x_test_ ,y_train_,y_test_= train_test_split(pca_normal,y_ori,test_size = 0.3,random_state =0 )

knn6 = KNeighborsClassifier(n_neighbors = k,weights='distance')
knn6.fit(x_train_,y_train_)
print("accuracy of 6-nearest neighbors : by PCA 2  normalization",knn6.score(x_test_,y_test_))
pc = knn6.score(x_test_,y_test_)
k6_pca_ac.append(pc)
```

    accuracy of 6-nearest neighbors : by PCA 1 normal  0.9005847953216374
    accuracy of 6-nearest neighbors : by PCA 9 scaled 0.9590643274853801
    accuracy of 6-nearest neighbors : by PCA min max 9  0.9707602339181286
    accuracy of 6-nearest neighbors : by PCA max abs 8  0.9473684210526315
    accuracy of 6-nearest neighbors : by PCA 2  normalization 0.9064327485380117
    


```python
accuracy_table_knn = pd.DataFrame()
accuracy_table_knn[" k= 1"] = k1_ac
accuracy_table_knn[" k= 2"] = k2_ac
accuracy_table_knn[" k= 3"] = k3_ac
accuracy_table_knn[" k= 4"] = k4_ac
accuracy_table_knn[" k= 5"] = k5_ac
accuracy_table_knn[" k= 6"] = k6_ac
accuracy_table_knn.index = ["ori","scale","minmax","max abs","normalize"]

accuracy_table_knn_pca = pd.DataFrame()
accuracy_table_knn_pca[" k= 1"] = k1_pca_ac
accuracy_table_knn_pca[" k= 2"] = k2_pca_ac
accuracy_table_knn_pca[" k= 3"] = k3_pca_ac
accuracy_table_knn_pca[" k= 4"] = k4_pca_ac
accuracy_table_knn_pca[" k= 5"] = k5_pca_ac
accuracy_table_knn_pca[" k= 6"] = k6_pca_ac
accuracy_table_knn_pca.index = ["ori","scale","minmax","max abs","normalize"]
```


```python
accuracy_table_knn
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>k= 1</th>
      <th>k= 2</th>
      <th>k= 3</th>
      <th>k= 4</th>
      <th>k= 5</th>
      <th>k= 6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ori</th>
      <td>0.918129</td>
      <td>0.918129</td>
      <td>0.923977</td>
      <td>0.923977</td>
      <td>0.947368</td>
      <td>0.941520</td>
    </tr>
    <tr>
      <th>scale</th>
      <td>0.935673</td>
      <td>0.935673</td>
      <td>0.947368</td>
      <td>0.953216</td>
      <td>0.959064</td>
      <td>0.964912</td>
    </tr>
    <tr>
      <th>minmax</th>
      <td>0.929825</td>
      <td>0.929825</td>
      <td>0.959064</td>
      <td>0.959064</td>
      <td>0.970760</td>
      <td>0.970760</td>
    </tr>
    <tr>
      <th>max abs</th>
      <td>0.947368</td>
      <td>0.947368</td>
      <td>0.941520</td>
      <td>0.947368</td>
      <td>0.953216</td>
      <td>0.953216</td>
    </tr>
    <tr>
      <th>normalize</th>
      <td>0.859649</td>
      <td>0.859649</td>
      <td>0.883041</td>
      <td>0.888889</td>
      <td>0.923977</td>
      <td>0.929825</td>
    </tr>
  </tbody>
</table>
</div>




```python
accuracy_table_knn_pca
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>k= 1</th>
      <th>k= 2</th>
      <th>k= 3</th>
      <th>k= 4</th>
      <th>k= 5</th>
      <th>k= 6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ori</th>
      <td>0.836257</td>
      <td>0.836257</td>
      <td>0.888889</td>
      <td>0.894737</td>
      <td>0.906433</td>
      <td>0.900585</td>
    </tr>
    <tr>
      <th>scale</th>
      <td>0.941520</td>
      <td>0.941520</td>
      <td>0.959064</td>
      <td>0.953216</td>
      <td>0.959064</td>
      <td>0.959064</td>
    </tr>
    <tr>
      <th>minmax</th>
      <td>0.941520</td>
      <td>0.941520</td>
      <td>0.959064</td>
      <td>0.959064</td>
      <td>0.964912</td>
      <td>0.970760</td>
    </tr>
    <tr>
      <th>max abs</th>
      <td>0.947368</td>
      <td>0.947368</td>
      <td>0.947368</td>
      <td>0.953216</td>
      <td>0.953216</td>
      <td>0.947368</td>
    </tr>
    <tr>
      <th>normalize</th>
      <td>0.888889</td>
      <td>0.888889</td>
      <td>0.894737</td>
      <td>0.894737</td>
      <td>0.906433</td>
      <td>0.906433</td>
    </tr>
  </tbody>
</table>
</div>




```python
accuracy_table_knn_pca==accuracy_table_knn
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>k= 1</th>
      <th>k= 2</th>
      <th>k= 3</th>
      <th>k= 4</th>
      <th>k= 5</th>
      <th>k= 6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ori</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>scale</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>minmax</th>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>max abs</th>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>normalize</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
# perceptron

from sklearn.datasets import load_digits
from sklearn.linear_model import Perceptron

x_train_, x_test_,y_train_,y_test_ = train_test_split(x_ori,y_ori,test_size = 0.3,random_state =0 )
perceptron = Perceptron(random_state=0).fit(x_train_, y_train_.values.ravel())

perceptron.score(x_test_, y_test_.values.ravel())

```

    C:\Users\Lee Joo Ye\Anaconda3\lib\site-packages\sklearn\linear_model\stochastic_gradient.py:128: FutureWarning: max_iter and tol parameters have been added in <class 'sklearn.linear_model.perceptron.Perceptron'> in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
      "and default tol will be 1e-3." % type(self), FutureWarning)
    




    0.9064327485380117




```python
# perceptron
perceptron_ac = []

from sklearn.datasets import load_digits
from sklearn.linear_model import Perceptron

################################### original ################################
x_train_, x_test_,y_train_,y_test_ = train_test_split(x_ori,y_ori,test_size = 0.3,random_state =0 )
perceptron = Perceptron(random_state=0).fit(x_train_, y_train_.values.ravel())
a = perceptron.score(x_test_, y_test_.values.ravel())
perceptron_ac.append(a)


################ scaled ##################
x_train_, x_test_,y_train_,y_test_ = train_test_split(scale_x,y_ori,test_size = 0.3,random_state =0 )
perceptron = Perceptron(random_state=0).fit(x_train_, y_train_.values.ravel())
a = perceptron.score(x_test_, y_test_.values.ravel())
perceptron_ac.append(a)

################## min_max_x ########################
x_train_, x_test_,y_train_,y_test_ = train_test_split(min_max_x,y_ori,test_size = 0.3,random_state =0 )
perceptron = Perceptron(random_state=0).fit(x_train_, y_train_.values.ravel())

a = perceptron.score(x_test_, y_test_.values.ravel())
perceptron_ac.append(a)

#####################  max_abs_x #####################
x_train_, x_test_,y_train_,y_test_ = train_test_split(max_abs_x,y_ori,test_size = 0.3,random_state =0 )
perceptron = Perceptron(random_state=0).fit(x_train_, y_train_.values.ravel())

a = perceptron.score(x_test_, y_test_.values.ravel())
perceptron_ac.append(a)

#################### normalization_x ################

x_train_, x_test_,y_train_,y_test_ = train_test_split(normalization_x,y_ori,test_size = 0.3,random_state =0 )
perceptron = Perceptron(random_state=0).fit(x_train_, y_train_.values.ravel())

a = perceptron.score(x_test_, y_test_.values.ravel())
perceptron_ac.append(a)
```


```python
#perceptron PCA

perceptron_pca_ac=[]

######################## ori ####################

x_train_, x_test_,y_train_,y_test_ = train_test_split(pca_norm,y_ori,test_size = 0.3,random_state =0 )
perceptron = Perceptron(random_state=0).fit(x_train_, y_train_.values.ravel())
a = perceptron.score(x_test_, y_test_.values.ravel())
perceptron_pca_ac.append(a)


##################### scaled ##################
x_train_, x_test_,y_train_,y_test_ = train_test_split(pca_scale ,y_ori,test_size = 0.3,random_state =0 )
perceptron = Perceptron(random_state=0).fit(x_train_, y_train_.values.ravel())
a = perceptron.score(x_test_, y_test_.values.ravel())
perceptron_pca_ac.append(a)

################## min_max_x ########################
x_train_, x_test_,y_train_,y_test_ = train_test_split(pca_minmax ,y_ori,test_size = 0.3,random_state =0 )
perceptron = Perceptron(random_state=0).fit(x_train_, y_train_.values.ravel())

a = perceptron.score(x_test_, y_test_.values.ravel())
perceptron_pca_ac.append(a)

#####################  max_abs_x #####################
x_train_, x_test_,y_train_,y_test_ = train_test_split(pca_maxabs ,y_ori,test_size = 0.3,random_state =0 )
perceptron = Perceptron(random_state=0).fit(x_train_, y_train_.values.ravel())

a = perceptron.score(x_test_, y_test_.values.ravel())
perceptron_pca_ac.append(a)

#################### normalization_x ################

x_train_, x_test_,y_train_,y_test_ = train_test_split(pca_normal ,y_ori,test_size = 0.3,random_state =0 )
perceptron = Perceptron(random_state=0).fit(x_train_, y_train_.values.ravel())

a = perceptron.score(x_test_, y_test_.values.ravel())
perceptron_pca_ac.append(a)
```


```python
p_ac = pd.DataFrame()
p_ac["perceptron"]=perceptron_ac
p_ac["perceptron_PCA"]=perceptron_pca_ac
p_ac.index = ["ori","scale","minmax","max abs","normalize"]
```


```python
from sklearn.neural_network import MLPClassifier

activation=('identity', 'logistic', 'tanh', 'relu')
solver = ['lbfgs', 'sgd','adam']


################################# original ###################################
mlp_acccuracy = pd.DataFrame()

x_train_, x_test_,y_train_,y_test_ =  train_test_split(x_ori,y_ori,test_size = 0.3,random_state =0 )

for i in solver:
    mlp_relu = MLPClassifier(solver = i, random_state=1, max_iter=10000).fit(x_train_, y_train)
    mlp_tanh = MLPClassifier(solver = i,random_state=1,activation='tanh', max_iter=10000).fit(x_train_, y_train_.values.ravel())
    mlp_i = MLPClassifier(solver = i,random_state=1,activation='identity', max_iter=10000).fit(x_train_, y_train_.values.ravel())
    mlp_l = MLPClassifier(solver = i,random_state=1,activation='logistic', max_iter=10000).fit(x_train_, y_train_.values.ravel())
    
    mlp_acccuracy[i] = [mlp_i.score(x_test_, y_test_.values.ravel()),mlp_l.score(x_test_, y_test_.values.ravel()),mlp_tanh.score(x_test_, y_test_.values.ravel()),mlp_relu.score(x_test_, y_test_.values.ravel()) ]
mlp_acccuracy.index=activation  
    
################################### scale ########################################
mlp_scale_acccuracy = pd.DataFrame()

x_train_, x_test_,y_train_,y_test_ =  train_test_split(scale_x,y_ori,test_size = 0.3,random_state =0 )

for i in solver:
    mlp_relu = MLPClassifier(solver = i, random_state=1, max_iter=10000).fit(x_train, y_train)
    mlp_tanh = MLPClassifier(solver = i,random_state=1,activation='tanh', max_iter=10000).fit(x_train_, y_train_.values.ravel())
    mlp_i = MLPClassifier(solver = i,random_state=1,activation='identity', max_iter=10000).fit(x_train_, y_train_.values.ravel())
    mlp_l = MLPClassifier(solver = i,random_state=1,activation='logistic', max_iter=10000).fit(x_train_, y_train_.values.ravel())
    
    mlp_scale_acccuracy[i] = [mlp_i.score(x_test_, y_test_.values.ravel()),mlp_l.score(x_test_, y_test_.values.ravel()),mlp_tanh.score(x_test_, y_test_.values.ravel()),mlp_relu.score(x_test_, y_test_.values.ravel()) ]
mlp_scale_acccuracy.index=activation   



########################################## min max #########################################
mlp_minmax_acccuracy = pd.DataFrame()

x_train_, x_test_,y_train_,y_test_ =  train_test_split(min_max_x,y_ori,test_size = 0.3,random_state =0 )

for i in solver:
    mlp_relu = MLPClassifier(solver = i, random_state=1, max_iter=10000).fit(x_train, y_train)
    mlp_tanh = MLPClassifier(solver = i,random_state=1,activation='tanh', max_iter=10000).fit(x_train_, y_train_.values.ravel())
    mlp_i = MLPClassifier(solver = i,random_state=1,activation='identity', max_iter=10000).fit(x_train_, y_train_.values.ravel())
    mlp_l = MLPClassifier(solver = i,random_state=1,activation='logistic', max_iter=10000).fit(x_train_, y_train_.values.ravel())
    
    mlp_minmax_acccuracy[i] = [mlp_i.score(x_test_, y_test_.values.ravel()),mlp_l.score(x_test_, y_test_.values.ravel()),mlp_tanh.score(x_test_, y_test_.values.ravel()),mlp_relu.score(x_test_, y_test_.values.ravel()) ]
mlp_minmax_acccuracy.index=activation   

######################################### max abs ##################################
mlp_maxabs_acccuracy = pd.DataFrame()

x_train_, x_test_,y_train_,y_test_ =  train_test_split(max_abs_x,y_ori,test_size = 0.3,random_state =0 )

for i in solver:
    mlp_relu = MLPClassifier(solver = i, random_state=1, max_iter=10000).fit(x_train, y_train)
    mlp_tanh = MLPClassifier(solver = i,random_state=1,activation='tanh', max_iter=10000).fit(x_train_, y_train_.values.ravel())
    mlp_i = MLPClassifier(solver = i,random_state=1,activation='identity', max_iter=10000).fit(x_train_, y_train_.values.ravel())
    mlp_l = MLPClassifier(solver = i,random_state=1,activation='logistic', max_iter=10000).fit(x_train_, y_train_.values.ravel())
    
    mlp_maxabs_acccuracy[i] = [mlp_i.score(x_test_, y_test_.values.ravel()),mlp_l.score(x_test_, y_test_.values.ravel()),mlp_tanh.score(x_test_, y_test_.values.ravel()),mlp_relu.score(x_test_, y_test_.values.ravel()) ]
mlp_maxabs_acccuracy.index=activation   


##################################### normalize ##############################

mlp_noramlize_acccuracy = pd.DataFrame()

x_train_, x_test_,y_train_,y_test_ =  train_test_split(normalization_x,y_ori,test_size = 0.3,random_state =0 )

for i in solver:
    mlp_relu = MLPClassifier(solver = i, random_state=1, max_iter=10000).fit(x_train, y_train)
    mlp_tanh = MLPClassifier(solver = i,random_state=1,activation='tanh', max_iter=10000).fit(x_train_, y_train_.values.ravel())
    mlp_i = MLPClassifier(solver = i,random_state=1,activation='identity', max_iter=10000).fit(x_train_, y_train_.values.ravel())
    mlp_l = MLPClassifier(solver = i,random_state=1,activation='logistic', max_iter=10000).fit(x_train_, y_train_.values.ravel())
    
    mlp_noramlize_acccuracy[i] = [mlp_i.score(x_test_, y_test_.values.ravel()),mlp_l.score(x_test_, y_test_.values.ravel()),mlp_tanh.score(x_test_, y_test_.values.ravel()),mlp_relu.score(x_test_, y_test_.values.ravel()) ]
mlp_noramlize_acccuracy.index=activation  

mlp_ac = pd.concat([mlp_acccuracy,mlp_scale_acccuracy,mlp_minmax_acccuracy,mlp_maxabs_acccuracy,mlp_noramlize_acccuracy],axis = 1)
mlp_ac.columns = [["normal"]*3+["scaled"]*3+["minmax"]*3+["maxmin"]*3+["normalized"]*3,['lbfgs', 'sgd','adam']*5 ]
mlp_ac.index = ['identity', 'logistic', 'tanh', 'relu']

```


```python
mlp_ac
```


```python
from sklearn.neural_network import MLPClassifier

activation=('identity', 'logistic', 'tanh', 'relu')
solver = ['lbfgs', 'sgd','adam']

# solver
######################################### original ####################################
mlp_acccuracy_pca = pd.DataFrame()
x_train_, x_test_,y_train_,y_test_ =  train_test_split(pca_norm,y_ori,test_size = 0.3,random_state =0 )

for i in solver:
    mlp_relu = MLPClassifier(solver = i, random_state=1, max_iter=10000).fit(x_train_, y_train)
    mlp_tanh = MLPClassifier(solver = i,random_state=1,activation='tanh', max_iter=10000).fit(x_train_, y_train_.values.ravel())
    mlp_i = MLPClassifier(solver = i,random_state=1,activation='identity', max_iter=10000).fit(x_train_, y_train_.values.ravel())
    mlp_l = MLPClassifier(solver = i,random_state=1,activation='logistic', max_iter=10000).fit(x_train_, y_train_.values.ravel())
    
    mlp_acccuracy_pca[i] = [mlp_i.score(x_test_, y_test_.values.ravel()),mlp_l.score(x_test_, y_test_.values.ravel()),mlp_tanh.score(x_test_, y_test_.values.ravel()),mlp_relu.score(x_test_, y_test_.values.ravel()) ]

############################################# scaled #####################################
mlp_acccuracy_pca_scaled = pd.DataFrame()
x_train_, x_test_,y_train_,y_test_ =  train_test_split(pca_scale,y_ori,test_size = 0.3,random_state =0 )

for i in solver:
    mlp_relu = MLPClassifier(solver = i, random_state=1, max_iter=10000).fit(x_train_, y_train)
    mlp_tanh = MLPClassifier(solver = i,random_state=1,activation='tanh', max_iter=10000).fit(x_train_, y_train_.values.ravel())
    mlp_i = MLPClassifier(solver = i,random_state=1,activation='identity', max_iter=10000).fit(x_train_, y_train_.values.ravel())
    mlp_l = MLPClassifier(solver = i,random_state=1,activation='logistic', max_iter=10000).fit(x_train_, y_train_.values.ravel())
    
    mlp_acccuracy_pca_scaled[i] = [mlp_i.score(x_test_, y_test_.values.ravel()),mlp_l.score(x_test_, y_test_.values.ravel()),mlp_tanh.score(x_test_, y_test_.values.ravel()),mlp_relu.score(x_test_, y_test_.values.ravel()) ]

############################################# min max #####################################  
mlp_acccuracy_pca_minmax = pd.DataFrame()
x_train_, x_test_,y_train_,y_test_ =  train_test_split(pca_minmax,y_ori,test_size = 0.3,random_state =0 )

for i in solver:
    mlp_relu = MLPClassifier(solver = i, random_state=1, max_iter=10000).fit(x_train_, y_train)
    mlp_tanh = MLPClassifier(solver = i,random_state=1,activation='tanh', max_iter=10000).fit(x_train_, y_train_.values.ravel())
    mlp_i = MLPClassifier(solver = i,random_state=1,activation='identity', max_iter=10000).fit(x_train_, y_train_.values.ravel())
    mlp_l = MLPClassifier(solver = i,random_state=1,activation='logistic', max_iter=10000).fit(x_train_, y_train_.values.ravel())
    
    mlp_acccuracy_pca_minmax[i] = [mlp_i.score(x_test_, y_test_.values.ravel()),mlp_l.score(x_test_, y_test_.values.ravel()),mlp_tanh.score(x_test_, y_test_.values.ravel()),mlp_relu.score(x_test_, y_test_.values.ravel()) ]

    
############################################# max abs #####################################  
mlp_acccuracy_pca_maxabs = pd.DataFrame()
x_train_, x_test_,y_train_,y_test_ =  train_test_split(pca_maxabs,y_ori,test_size = 0.3,random_state =0 )

for i in solver:
    mlp_relu = MLPClassifier(solver = i, random_state=1, max_iter=10000).fit(x_train_, y_train)
    mlp_tanh = MLPClassifier(solver = i,random_state=1,activation='tanh', max_iter=10000).fit(x_train_, y_train_.values.ravel())
    mlp_i = MLPClassifier(solver = i,random_state=1,activation='identity', max_iter=10000).fit(x_train_, y_train_.values.ravel())
    mlp_l = MLPClassifier(solver = i,random_state=1,activation='logistic', max_iter=10000).fit(x_train_, y_train_.values.ravel())
    
    mlp_acccuracy_pca_maxabs[i] = [mlp_i.score(x_test_, y_test_.values.ravel()),mlp_l.score(x_test_, y_test_.values.ravel()),mlp_tanh.score(x_test_, y_test_.values.ravel()),mlp_relu.score(x_test_, y_test_.values.ravel()) ]
    
############################################# normalized #####################################  
mlp_acccuracy_pca_normal = pd.DataFrame()
x_train_, x_test_,y_train_,y_test_ =  train_test_split(pca_normal,y_ori,test_size = 0.3,random_state =0 )

for i in solver:
    mlp_relu = MLPClassifier(solver = i, random_state=1, max_iter=10000).fit(x_train_, y_train)
    mlp_tanh = MLPClassifier(solver = i,random_state=1,activation='tanh', max_iter=10000).fit(x_train_, y_train_.values.ravel())
    mlp_i = MLPClassifier(solver = i,random_state=1,activation='identity', max_iter=10000).fit(x_train_, y_train_.values.ravel())
    mlp_l = MLPClassifier(solver = i,random_state=1,activation='logistic', max_iter=10000).fit(x_train_, y_train_.values.ravel())
    
    mlp_acccuracy_pca_normal[i] = [mlp_i.score(x_test_, y_test_.values.ravel()),mlp_l.score(x_test_, y_test_.values.ravel()),mlp_tanh.score(x_test_, y_test_.values.ravel()),mlp_relu.score(x_test_, y_test_.values.ravel()) ]
 
```


```python
mlpl_pca = pd.concat( [mlp_acccuracy_pca, mlp_acccuracy_pca_sclaed, mlp_acccuracy_pca_minmax, mlp_acccuracy_pca_maxabs, mlp_acccuracy_pca_normal] , axis = 1)
mlpl_pca.columns = [["normal"]*3+["scaled"]*3+["minmax"]*3+["maxmin"]*3+["normalized"]*3,['lbfgs', 'sgd','adam']*5 ]
mlpl_pca.index = ['identity', 'logistic', 'tanh', 'relu']
```


```python
mlpl_pca
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="3" halign="left">normal</th>
      <th colspan="3" halign="left">scaled</th>
      <th colspan="3" halign="left">minmax</th>
      <th colspan="3" halign="left">maxmin</th>
      <th colspan="3" halign="left">normalized</th>
    </tr>
    <tr>
      <th></th>
      <th>lbfgs</th>
      <th>sgd</th>
      <th>adam</th>
      <th>lbfgs</th>
      <th>sgd</th>
      <th>adam</th>
      <th>lbfgs</th>
      <th>sgd</th>
      <th>adam</th>
      <th>lbfgs</th>
      <th>sgd</th>
      <th>adam</th>
      <th>lbfgs</th>
      <th>sgd</th>
      <th>adam</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>identity</th>
      <td>0.912281</td>
      <td>0.923977</td>
      <td>0.923977</td>
      <td>0.941520</td>
      <td>0.959064</td>
      <td>0.959064</td>
      <td>0.970760</td>
      <td>0.941520</td>
      <td>0.976608</td>
      <td>0.970760</td>
      <td>0.941520</td>
      <td>0.970760</td>
      <td>0.941520</td>
      <td>0.918129</td>
      <td>0.941520</td>
    </tr>
    <tr>
      <th>logistic</th>
      <td>0.923977</td>
      <td>0.918129</td>
      <td>0.906433</td>
      <td>0.935673</td>
      <td>0.941520</td>
      <td>0.959064</td>
      <td>0.970760</td>
      <td>0.631579</td>
      <td>0.976608</td>
      <td>0.964912</td>
      <td>0.631579</td>
      <td>0.970760</td>
      <td>0.877193</td>
      <td>0.631579</td>
      <td>0.935673</td>
    </tr>
    <tr>
      <th>tanh</th>
      <td>0.918129</td>
      <td>0.923977</td>
      <td>0.912281</td>
      <td>0.959064</td>
      <td>0.959064</td>
      <td>0.964912</td>
      <td>0.970760</td>
      <td>0.941520</td>
      <td>0.976608</td>
      <td>0.964912</td>
      <td>0.941520</td>
      <td>0.970760</td>
      <td>0.888889</td>
      <td>0.918129</td>
      <td>0.941520</td>
    </tr>
    <tr>
      <th>relu</th>
      <td>0.912281</td>
      <td>0.923977</td>
      <td>0.923977</td>
      <td>0.964912</td>
      <td>0.941520</td>
      <td>0.976608</td>
      <td>0.964912</td>
      <td>0.929825</td>
      <td>0.976608</td>
      <td>0.976608</td>
      <td>0.935673</td>
      <td>0.976608</td>
      <td>0.923977</td>
      <td>0.918129</td>
      <td>0.941520</td>
    </tr>
  </tbody>
</table>
</div>


