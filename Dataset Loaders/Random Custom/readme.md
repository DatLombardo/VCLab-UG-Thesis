Random Custom TVSum50 Dataset
=====================

### Contributors  
* Michael Lombardo
* Faisal Qureshi


1 &nbsp;&nbsp;&nbsp;&nbsp;Introduction
============

After creating a TVSum50 dataset loader, it was clear that determining
boundaries of the dataset would be extremely difficult due to segments
being 30 consecutive frames without clear cuts for boundaries. This new
custom dataset of TVSum50 takes videos from TVSum50 and merges two
seperate videos together if there is a boundary within the 9 frame
segment, the segment will be found anywhere between the second and
last frame. If no boundary is present the segment contains only one
video throughout all 9 frames.

During training the segment is broken into short sequences of three
frames so the LSTM does not use a large amount of memory. It is
typical in scholary work that given networks are trained on sequence
lengths of 30+, this will be explored at a later date. Currently the sequences were broken up as follows: <br>
[[0 1 2], <br>
 [1 2 3], <br>
  ...     <br>
 [6 7 8]]

2 &nbsp;&nbsp;&nbsp;&nbsp;Dependencies
============

* **numpy** : v1.15.3, Default matrix generation & array manipulation.
* **random** : v3.2, Generation of boundary values, segment start points.
* **torch** : v0.4.1, Processing tensors, saving & loading pytorch tensor files of segments.
* **csv** : v1.0, Reading & writing csv files.
* **cv2** : v3.4.3, Video Capture functions.

3 &nbsp;&nbsp;&nbsp;&nbsp;Results
============
Running Over a Single Data Item
-----
The model was ran over a single data item which is 9x3x224x224,
then expressed into sequences of 3, such that the data item was
**7x3x3x224x224**
<details><summary>Results Here</summary>
<p>

batch: 0, epoch:99, error: 0.01907660998404026 <br>
batch: 1, epoch:99, error: 0.09815434366464615<br>
batch: 0, epoch:199, error: 0.006939364597201347<br>
batch: 1, epoch:199, error: 0.04145383462309837<br>
batch: 0, epoch:299, error: 0.003786869579926133<br>
batch: 1, epoch:299, error: 0.024179920554161072<br>
batch: 0, epoch:399, error: 0.0024631733540445566<br>
batch: 1, epoch:399, error: 0.016475912183523178<br>
batch: 0, epoch:499, error: 0.0018162851920351386<br>
batch: 1, epoch:499, error: 0.012346982955932617<br>
batch: 0, epoch:599, error: 0.0014303672360256314<br>
batch: 1, epoch:599, error: 0.0098041370511055<br>
batch: 0, epoch:699, error: 0.0011693421984091401<br>
batch: 1, epoch:699, error: 0.008074057288467884<br>
batch: 0, epoch:799, error: 0.0009820455452427268<br>
batch: 1, epoch:799, error: 0.0068256244994699955<br>
batch: 0, epoch:899, error: 0.0008444422273896635<br>
batch: 1, epoch:899, error: 0.0058973548002541065<br>
batch: 0, epoch:999, error: 0.0007404612260870636<br>
batch: 1, epoch:999, error: 0.0051864939741790295<br>

</p>
</details>

Running Over Twelve Data Items
-----
The model was ran over a twelve data items which is 9x3x224x224,
then expressed into sequences of 3, such that the testing data was
**12x7x3x3x224x224**
<details><summary>Results Here</summary>
<p>

item: 1 epoch:0
<br>	batch: 0, error: 0.6411216259002686
<br>item: 1 epoch:0
<br>	batch: 1, error: 0.371522456407547
<br>item: 2 epoch:0
<br>	batch: 0, error: 0.5897281765937805
<br>item: 2 epoch:0
<br>	batch: 1, error: 0.34323057532310486
<br>item: 3 epoch:0
<br>	batch: 0, error: 0.5986451506614685
<br>item: 3 epoch:0
<br>	batch: 1, error: 0.6177887320518494
<br>item: 4 epoch:0
<br>	batch: 0, error: 0.6001524925231934
<br>item: 4 epoch:0
<br>	batch: 1, error: 0.42257678508758545
<br>item: 5 epoch:0
<br>	batch: 0, error: 0.646174967288971
<br>item: 5 epoch:0
<br>	batch: 1, error: 0.6752721667289734
<br>item: 6 epoch:0
<br>	batch: 0, error: 0.6160426735877991
<br>item: 6 epoch:0
<br>	batch: 1, error: 0.6327351927757263
<br>item: 7 epoch:0
<br>	batch: 0, error: 0.595893383026123
<br>item: 7 epoch:0
<br>	batch: 1, error: 0.5786117911338806
<br>item: 8 epoch:0
<br>	batch: 0, error: 0.6208403706550598
<br>item: 8 epoch:0
<br>	batch: 1, error: 0.565699577331543
<br>item: 9 epoch:0
<br>	batch: 0, error: 0.5800976157188416
<br>item: 9 epoch:0
<br>	batch: 1, error: 0.34943708777427673
<br>item: 10 epoch:0
<br>	batch: 0, error: 0.5246416926383972
<br>item: 10 epoch:0
<br>	batch: 1, error: 0.41872987151145935
<br>item: 11 epoch:0
<br>	batch: 0, error: 0.6562676429748535
<br>item: 11 epoch:0
<br>	batch: 1, error: 0.48437002301216125
<br>item: 12 epoch:0
<br>	batch: 0, error: 0.678760826587677
<br>item: 12 epoch:0
<br>	batch: 1, error: 0.44569721817970276
<br>item: 1 epoch:20
<br>	batch: 0, error: 0.0064130802638828754
<br>item: 1 epoch:20
<br>	batch: 1, error: 0.006138892378658056
<br>item: 2 epoch:20
<br>	batch: 0, error: 0.005167619790881872
<br>item: 2 epoch:20
<br>	batch: 1, error: 0.005319856107234955
<br>item: 3 epoch:20
<br>	batch: 0, error: 0.018834900110960007
<br>item: 3 epoch:20
<br>	batch: 1, error: 0.18362492322921753
<br>item: 4 epoch:20
<br>	batch: 0, error: 0.014552640728652477
<br>item: 4 epoch:20
<br>	batch: 1, error: 0.008557094261050224
<br>item: 5 epoch:20
<br>	batch: 0, error: 0.06683100014925003
<br>item: 5 epoch:20
<br>	batch: 1, error: 0.2134353667497635
<br>item: 6 epoch:20
<br>	batch: 0, error: 0.250338613986969
<br>item: 6 epoch:20
<br>	batch: 1, error: 0.1720888614654541
<br>item: 7 epoch:20
<br>	batch: 0, error: 0.01758456788957119
<br>item: 7 epoch:20
<br>	batch: 1, error: 0.14159531891345978
<br>item: 8 epoch:20
	<br> batch: 0, error: 0.04310928285121918
<br> item: 8 epoch:20
	<br> batch: 1, error: 0.2355043888092041
<br> item: 9 epoch:20
	<br> batch: 0, error: 0.004700371529906988
<br> item: 9 epoch:20
	<br> batch: 1, error: 0.004547682125121355
<br> item: 10 epoch:20
	<br> batch: 0, error: 0.012052323669195175
<br> item: 10 epoch:20
	<br> batch: 1, error: 0.06343140453100204
<br> item: 11 epoch:20
	<br> batch: 0, error: 0.18464058637619019
<br> item: 11 epoch:20
	<br> batch: 1, error: 0.11330071091651917
<br> item: 12 epoch:20
	<br> batch: 0, error: 0.2870674431324005
<br> item: 12 epoch:20
	<br> batch: 1, error: 0.20932620763778687
<br> item: 1 epoch:40
	<br> batch: 0, error: 0.0016205409774556756
<br> item: 1 epoch:40
	<br> batch: 1, error: 0.0015435254899784923
<br> item: 2 epoch:40
	<br> batch: 0, error: 0.0012532040709629655
<br> item: 2 epoch:40
	<br> batch: 1, error: 0.0012706852285191417
<br> item: 3 epoch:40
	<br> batch: 0, error: 0.005810637027025223
<br> item: 3 epoch:40
	<br> batch: 1, error: 0.08419737964868546
<br> item: 4 epoch:40
	<br> batch: 0, error: 0.0044464548118412495
<br> item: 4 epoch:40
	<br> batch: 1, error: 0.002152708824723959
<br> item: 5 epoch:40
	<br> batch: 0, error: 0.021339938044548035
<br> item: 5 epoch:40
	<br> batch: 1, error: 0.09446415305137634
<br> item: 6 epoch:40
	<br> batch: 0, error: 0.10644194483757019
<br> item: 6 epoch:40
	<br> batch: 1, error: 0.08249013125896454
<br> item: 7 epoch:40
	<br> batch: 0, error: 0.004868895281106234
<br> item: 7 epoch:40
	<br> batch: 1, error: 0.055371467024087906
<br> item: 8 epoch:40
	<br> batch: 0, error: 0.013547119684517384
<br> item: 8 epoch:40
	<br> batch: 1, error: 0.14471353590488434
<br> item: 9 epoch:40
	<br> batch: 0, error: 0.0009818322723731399
<br> item: 9 epoch:40
	<br> batch: 1, error: 0.0009205093956552446
<br> item: 10 epoch:40
	<br> batch: 0, error: 0.0035303914919495583
<br> item: 10 epoch:40
	<br> batch: 1, error: 0.018655790016055107
<br> item: 11 epoch:40
	<br> batch: 0, error: 0.07510904222726822
<br> item: 11 epoch:40
	<br> batch: 1, error: 0.04914167895913124
<br> item: 12 epoch:40
	<br> batch: 0, error: 0.17533379793167114
<br> item: 12 epoch:40
	<br> batch: 1, error: 0.15058505535125732
<br> item: 1 epoch:60
	<br> batch: 0, error: 0.0006963122286833823
<br> item: 1 epoch:60
	<br> batch: 1, error: 0.0006611947901546955
<br> item: 2 epoch:60
	<br> batch: 0, error: 0.0005355384782887995
<br> item: 2 epoch:60
	<br> batch: 1, error: 0.0005325700039975345
<br> item: 3 epoch:60
	<br> batch: 0, error: 0.0028004180639982224
<br> item: 3 epoch:60
	<br> batch: 1, error: 0.05028463527560234
<br> item: 4 epoch:60
	<br> batch: 0, error: 0.002210229868069291
<br> item: 4 epoch:60
	<br> batch: 1, error: 0.0009274016483686864
<br> item: 5 epoch:60
	<br> batch: 0, error: 0.010414252988994122
<br> item: 5 epoch:60
	<br> batch: 1, error: 0.054287564009428024
<br> item: 6 epoch:60
	<br> batch: 0, error: 0.06061435863375664
<br> item: 6 epoch:60
	<br> batch: 1, error: 0.048176493495702744
<br> item: 7 epoch:60
	<br> batch: 0, error: 0.0022557496558874846
<br> item: 7 epoch:60
	<br> batch: 1, error: 0.0324421189725399
<br> item: 8 epoch:60
	<br> batch: 0, error: 0.006347940769046545
<br> item: 8 epoch:60
	<br> batch: 1, error: 0.09738075733184814
<br> item: 9 epoch:60
	<br> batch: 0, error: 0.0003942078910768032
<br> item: 9 epoch:60
	<br> batch: 1, error: 0.0003608020197134465
<br> item: 10 epoch:60
	<br> batch: 0, error: 0.0016841605538502336
<br> item: 10 epoch:60
	<br> batch: 1, error: 0.008916972205042839
<br> item: 11 epoch:60
	<br> batch: 0, error: 0.041769128292798996
<br> item: 11 epoch:60
	<br> batch: 1, error: 0.028168706223368645
<br> item: 12 epoch:60
	<br> batch: 0, error: 0.10102760791778564
<br> item: 12 epoch:60
	<br> batch: 1, error: 0.0812162384390831
<br> item: 1 epoch:80
	<br> batch: 0, error: 0.0003899137955158949
<br> item: 1 epoch:80
	<br> batch: 1, error: 0.0003684436669573188
<br> item: 2 epoch:80
	<br> batch: 0, error: 0.0003016758419107646
<br> item: 2 epoch:80
	<br> batch: 1, error: 0.00029185411403886974
<br> item: 3 epoch:80
	<br> batch: 0, error: 0.0016839135205373168
<br> item: 3 epoch:80
	<br> batch: 1, error: 0.03414633125066757
<br> item: 4 epoch:80
	<br> batch: 0, error: 0.0013565676053985953
<br> item: 4 epoch:80
	<br> batch: 1, error: 0.000506110314745456
<br> item: 5 epoch:80
	<br> batch: 0, error: 0.006286520045250654
<br> item: 5 epoch:80
	<br> batch: 1, error: 0.034980494529008865
<br> item: 6 epoch:80
	<br> batch: 0, error: 0.03917141631245613
<br> item: 6 epoch:80
	<br> batch: 1, error: 0.031602948904037476
<br> item: 7 epoch:80
	<br> batch: 0, error: 0.0013265716843307018
<br> item: 7 epoch:80
	<br> batch: 1, error: 0.02176518738269806
<br> item: 8 epoch:80
	<br> batch: 0, error: 0.00395030714571476
<br> item: 8 epoch:80
	<br> batch: 1, error: 0.0627729520201683
<br> item: 9 epoch:80
	<br> batch: 0, error: 0.00020923152624163777
<br> item: 9 epoch:80
	<br> batch: 1, error: 0.00018844536680262536
<br> item: 10 epoch:80
	<br> batch: 0, error: 0.0010080361971631646
<br> item: 10 epoch:80
	<br> batch: 1, error: 0.005367970559746027
<br> item: 11 epoch:80
	<br> batch: 0, error: 0.026437506079673767
<br> item: 11 epoch:80
	<br> batch: 1, error: 0.018431469798088074
<br> item: 12 epoch:80
	<br> batch: 0, error: 0.05819988623261452
<br> item: 12 epoch:80
	<br> batch: 1, error: 0.04667757824063301
<br> item: 1 epoch:100
	<br> batch: 0, error: 0.00025263632414862514
<br> item: 1 epoch:100
	<br> batch: 1, error: 0.0002378934295848012
<br> item: 2 epoch:100
	<br> batch: 0, error: 0.00019047832756768912
<br> item: 2 epoch:100
	<br> batch: 1, error: 0.0001826767693273723
<br> item: 3 epoch:100
	<br> batch: 0, error: 0.0011489217868074775
<br> item: 3 epoch:100
	<br> batch: 1, error: 0.025023648515343666
<br> item: 4 epoch:100
	<br> batch: 0, error: 0.0009325225837528706
<br> item: 4 epoch:100
	<br> batch: 1, error: 0.00032057942007668316
<br> item: 5 epoch:100
	<br> batch: 0, error: 0.004338028375059366
<br> item: 5 epoch:100
	<br> batch: 1, error: 0.02512536756694317
<br> item: 6 epoch:100
	<br> batch: 0, error: 0.02785199135541916
<br> item: 6 epoch:100
	<br> batch: 1, error: 0.022564997896552086
<br> item: 7 epoch:100
	<br> batch: 0, error: 0.0008804092067293823
<br> item: 7 epoch:100
	<br> batch: 1, error: 0.015615858137607574
<br> item: 8 epoch:100
	<br> batch: 0, error: 0.0027809245511889458
<br> item: 8 epoch:100
	<br> batch: 1, error: 0.04323406517505646
<br> item: 9 epoch:100
	<br> batch: 0, error: 0.00012995376891922206
<br> item: 9 epoch:100
	<br> batch: 1, error: 0.0001161515319836326
<br> item: 10 epoch:100
	<br> batch: 0, error: 0.0006806386518292129
<br> item: 10 epoch:100
	<br> batch: 1, error: 0.003673688741400838
<br> item: 11 epoch:100
	<br> batch: 0, error: 0.01822611689567566
<br> item: 11 epoch:100
	<br> batch: 1, error: 0.012907358817756176
<br> item: 12 epoch:100
	<br> batch: 0, error: 0.04117694869637489
<br> item: 12 epoch:100
	<br> batch: 1, error: 0.03335340693593025

</p>
</details>
