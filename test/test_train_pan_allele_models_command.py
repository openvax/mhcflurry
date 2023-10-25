"""
Tests for training and predicting using Class1 pan-allele models.
"""
from . import initialize
initialize()

import json
import os
import shutil
import tempfile
import subprocess

import pandas

from numpy.testing import assert_equal, assert_array_less

from mhcflurry import Class1AffinityPredictor
from mhcflurry.downloads import get_path

from mhcflurry.testing_utils import cleanup, startup
teardown = cleanup
setup = startup

os.environ["CUDA_VISIBLE_DEVICES"] = ""


HYPERPARAMETERS_LIST = [
{
    'activation': 'tanh',
    'allele_dense_layer_sizes': [],
    'batch_normalization': False,
    'dense_layer_l1_regularization': 0.0,
    'dense_layer_l2_regularization': 0.0,
    'dropout_probability': 0.5,
    'early_stopping': True,
    'init': 'glorot_uniform',
    'layer_sizes': [64],
    'learning_rate': None,
    'locally_connected_layers': [],
    'loss': 'custom:mse_with_inequalities',
    'max_epochs': 0,  # never selected
    'minibatch_size': 256,
    'optimizer': 'rmsprop',
    'output_activation': 'sigmoid',
    'patience': 10,
    'peptide_allele_merge_activation': '',
    'peptide_allele_merge_method': 'concatenate',
    'peptide_amino_acid_encoding': 'BLOSUM62',
    'peptide_dense_layer_sizes': [],
    'peptide_encoding': {
        'alignment_method': 'left_pad_centered_right_pad',
        'max_length': 15,
        'vector_encoding_name': 'BLOSUM62',
    },
    'random_negative_affinity_max': 50000.0,
    'random_negative_affinity_min': 20000.0,
    'random_negative_constant': 25,
    'random_negative_distribution_smoothing': 0.0,
    'random_negative_match_distribution': True,
    'random_negative_rate': 0.2,
    'train_data': {"pretrain": False},
    'validation_split': 0.1,
    'data_dependent_initialization_method': "lsuv",
},
{
    'activation': 'tanh',
    'allele_dense_layer_sizes': [],
    'batch_normalization': False,
    'dense_layer_l1_regularization': 0.0,
    'dense_layer_l2_regularization': 0.0,
    'dropout_probability': 0.5,
    'early_stopping': True,
    'init': 'glorot_uniform',
    'layer_sizes': [32],
    'learning_rate': None,
    'locally_connected_layers': [],
    'loss': 'custom:mse_with_inequalities',
    'max_epochs': 5,
    'minibatch_size': 256,
    'optimizer': 'rmsprop',
    'output_activation': 'sigmoid',
    'patience': 10,
    'peptide_allele_merge_activation': '',
    'peptide_allele_merge_method': 'concatenate',
    'peptide_amino_acid_encoding': 'BLOSUM62',
    'peptide_dense_layer_sizes': [],
    'peptide_encoding': {
        'alignment_method': 'left_pad_centered_right_pad',
        'max_length': 15,
        'vector_encoding_name': 'BLOSUM62',
    },
    'random_negative_affinity_max': 50000.0,
    'random_negative_affinity_min': 20000.0,
    'random_negative_constant': 25,
    'random_negative_distribution_smoothing': 0.0,
    'random_negative_match_distribution': True,
    'random_negative_rate': 0.2,
    'train_data': {
        "pretrain": True,
        'pretrain_peptides_per_step': 4,
        'pretrain_max_epochs': 2,
        'pretrain_max_val_loss': 0.2,
    },
    'validation_split': 0.1,
},
]

PRETRAIN_DATA = """
,BoLA-6*13:01,Eqca-1*01:01,H-2-Db,H-2-Dd,H-2-Kb,H-2-Kd,H-2-Kk,H-2-Ld,HLA-A*01:01,HLA-A*02:01,HLA-A*02:02,HLA-A*02:03,HLA-A*02:05,HLA-A*02:06,HLA-A*02:07,HLA-A*02:11,HLA-A*02:12,HLA-A*02:16,HLA-A*02:17,HLA-A*02:19,HLA-A*02:50,HLA-A*03:01,HLA-A*11:01,HLA-A*23:01,HLA-A*24:02,HLA-A*24:03,HLA-A*25:01,HLA-A*26:01,HLA-A*26:02,HLA-A*26:03,HLA-A*29:02,HLA-A*30:01,HLA-A*30:02,HLA-A*31:01,HLA-A*32:01,HLA-A*33:01,HLA-A*66:01,HLA-A*68:01,HLA-A*68:02,HLA-A*68:23,HLA-A*69:01,HLA-A*80:01,HLA-B*07:01,HLA-B*07:02,HLA-B*08:01,HLA-B*08:02,HLA-B*08:03,HLA-B*14:02,HLA-B*15:01,HLA-B*15:02,HLA-B*15:03,HLA-B*15:09,HLA-B*15:17,HLA-B*18:01,HLA-B*27:03,HLA-B*27:05,HLA-B*35:01,HLA-B*35:03,HLA-B*38:01,HLA-B*39:01,HLA-B*40:01,HLA-B*40:02,HLA-B*42:01,HLA-B*44:02,HLA-B*44:03,HLA-B*45:01,HLA-B*46:01,HLA-B*48:01,HLA-B*51:01,HLA-B*53:01,HLA-B*54:01,HLA-B*57:01,HLA-B*58:01,HLA-B*83:01,HLA-C*03:03,HLA-C*05:01,HLA-C*06:02,HLA-C*07:02,HLA-C*12:03,HLA-C*15:02,Mamu-A*01:01,Mamu-A*02:01,Mamu-A*02:0102,Mamu-A*07:01,Mamu-A*11:01,Mamu-A*22:01,Mamu-A*26:01,Mamu-B*01:01,Mamu-B*03:01,Mamu-B*08:01,Mamu-B*10:01,Mamu-B*17:01,Mamu-B*17:04,Mamu-B*39:01,Mamu-B*52:01,Mamu-B*83:01,Patr-A*01:01,Patr-A*04:01,Patr-A*07:01,Patr-A*09:01,Patr-B*01:01,Patr-B*13:01,Patr-B*24:01
VCCIYWDISYCTCQ,44182.5,42134.1,44152.5,42887.9,30441.3,38592.9,41962.6,42822.6,25616.4,22063.8,35987.9,36692.9,40859.1,41072.7,38672.0,36913.2,40334.9,36647.9,36738.3,43307.7,40456.6,29511.6,31662.5,42630.7,40073.5,42494.2,36130.4,39625.4,39909.6,38751.8,39681.3,36767.7,36100.5,29302.9,46286.3,26136.6,37672.1,34302.1,37976.5,37864.7,31506.7,32281.1,40357.5,35316.3,29220.1,35346.3,34818.9,41566.6,31951.6,38975.1,40825.3,44917.4,43148.3,34119.6,37508.6,37930.1,29881.9,42463.0,39552.8,36529.9,36587.0,35603.6,34376.4,37498.8,42253.7,37676.9,30077.2,39384.4,33218.6,36576.7,38777.8,29806.4,28986.1,34746.3,39748.3,38600.2,40338.9,39364.9,41546.0,39574.0,42059.6,46195.4,41975.4,44810.3,44629.7,42749.5,43288.1,46931.2,39014.4,39893.3,41316.9,41923.5,41918.2,42723.5,38612.0,39772.1,42175.5,38879.7,41908.9,38287.3,41522.6,39666.8,43396.1
CQFVANRCHQKVFRL,39146.5,39452.0,9484.7,39116.9,13002.1,27669.4,30010.9,37655.0,31624.9,12478.3,28164.5,30813.8,39656.3,24968.0,33852.1,30953.1,31049.5,32169.5,33800.9,36730.1,37233.0,17696.0,31413.6,26134.9,31618.0,40342.3,35511.9,38680.3,38605.3,37388.9,38012.0,36161.2,36449.0,23276.0,39330.3,38518.9,37164.7,36824.9,32292.1,38968.4,31223.9,31704.9,35919.0,32976.9,28484.1,33958.6,32564.3,37876.1,24948.2,36401.9,34953.6,36509.9,39256.5,28768.4,24109.5,10714.6,40389.5,43803.1,32800.9,26891.8,27654.3,23972.8,35700.0,32613.3,26999.3,28714.6,29744.2,33689.4,34232.0,32967.2,37510.6,32342.9,30111.8,30403.3,38445.6,38237.8,37861.1,37985.3,39969.8,39538.2,38738.5,42978.4,39435.6,41359.5,16024.0,39753.9,41649.9,39454.1,32748.2,27406.8,38950.9,30001.9,27833.5,32655.5,27208.1,37915.8,39650.7,37642.0,37611.6,32313.3,36946.5,33275.5,39516.1
YNWDWAQCSGI,35747.3,29140.6,30472.7,33365.7,1476.4,4321.6,10641.6,32903.8,20380.9,4780.3,12919.5,17551.4,22162.1,13392.5,16314.1,7991.3,23261.7,6525.2,15695.2,27362.5,24716.2,28711.7,30898.9,23808.7,25642.9,39880.0,30787.6,33162.7,33052.3,33146.9,21397.6,36392.2,26063.9,28349.8,40062.1,26596.2,32116.4,31183.4,12831.5,35000.6,17873.4,30601.4,30336.8,24523.8,18060.1,28888.9,26060.1,34612.6,14003.0,32129.0,31080.5,35535.7,32822.4,24605.3,20247.7,8311.8,31112.8,40916.8,29408.5,29217.6,31888.1,25260.8,27729.2,28121.0,36758.1,32991.8,25414.3,32947.6,6063.8,33594.4,20760.6,28579.3,25034.7,23129.8,35563.0,30887.8,31750.0,27229.8,33731.0,32884.0,5407.3,4834.1,35326.9,25099.6,3516.2,32488.6,32741.2,40440.2,20202.0,21406.5,31991.4,22185.6,24853.2,27460.7,5335.4,15766.5,34520.1,33168.8,32759.1,10026.4,3042.1,27291.3,17873.3
APEPVMMQGCDN,44794.5,42064.7,36712.6,42567.1,31312.9,25512.2,38772.9,32376.8,31503.8,28373.3,45393.5,41842.1,43223.5,45234.2,39579.4,40729.5,41441.1,38673.3,37791.6,43381.7,42993.3,34470.7,37888.9,43650.8,41233.2,42212.6,34542.5,37432.8,37800.1,36891.9,40060.0,40205.7,35137.4,33173.1,46841.4,41392.7,36465.8,37628.0,41854.1,39959.8,30890.1,32541.7,34548.9,20953.5,34911.7,37295.7,35305.7,37963.4,35794.0,39768.6,41915.4,44501.8,43077.3,34135.5,36039.6,36968.8,30841.4,40495.2,36776.4,34237.8,33980.3,34729.3,36210.8,34585.1,41600.5,36181.2,28528.2,38973.9,32572.4,37361.4,35637.2,31359.7,31602.5,24366.2,39960.9,36509.1,43038.1,41238.5,41575.8,37826.4,39195.7,45378.3,42078.6,44831.9,41156.4,39194.0,44144.7,46339.5,39007.9,40269.1,40479.5,41508.5,41613.3,42798.9,38927.9,37961.6,43384.7,40127.5,43129.2,38632.1,42726.0,31364.8,43992.3
SIQNDHQFCNE,41247.1,36778.2,31563.6,39917.4,10018.5,26242.4,31501.4,36002.1,28366.2,23209.0,28116.8,27058.3,33981.8,23981.3,33667.0,30816.8,36286.8,30787.3,33485.0,40625.7,32634.4,11132.3,2621.0,37879.7,31918.2,41386.8,30477.2,29618.2,30679.9,31896.3,36876.3,29016.4,21555.8,16172.6,35160.7,22537.3,32996.3,21159.4,27137.4,29441.1,29994.0,32416.6,33022.6,31767.9,19129.5,25393.9,30423.1,37177.4,17995.5,32992.6,33549.6,39935.4,43477.4,33433.2,34472.0,34017.7,41050.6,44500.3,37744.9,35526.9,34872.2,33849.1,26222.6,32226.4,42709.1,36014.3,26019.6,38845.3,32771.8,37961.9,40955.2,21375.2,28595.7,30962.6,38296.9,37179.4,40323.2,38384.1,39617.4,38508.9,33177.7,25294.9,37478.4,42231.1,30983.3,33685.5,39692.9,45876.2,32965.1,34191.3,33783.1,36999.0,37443.1,38511.6,21638.2,31981.6,35943.1,30735.1,37418.3,19187.4,35391.0,36495.6,37896.1
WAVYMCISAPL,26021.0,16088.8,5262.2,11881.8,3628.8,10658.3,7398.9,18827.6,24553.0,6062.3,3725.9,6626.5,12806.4,1919.8,21857.7,13408.4,16709.7,11225.4,21476.2,33110.7,10213.0,20997.3,25916.5,17285.9,14772.3,39728.6,14817.4,13098.4,17583.7,18495.2,8523.0,24610.3,16200.6,12117.8,29147.3,12013.4,21713.9,5327.3,340.7,5501.6,10100.1,31368.4,13169.4,2241.8,1718.9,9124.3,13725.7,15124.6,2411.5,15438.8,6023.0,13412.1,11652.5,16573.2,10559.6,4402.9,1293.3,19356.7,5939.7,4715.8,12976.0,10061.8,8468.5,8280.0,17922.2,10362.6,6628.0,11632.8,8235.4,9815.9,4647.5,14010.5,7109.3,5989.7,4340.4,5718.2,22291.9,12581.6,19864.1,6913.5,2876.7,249.7,23174.7,7008.9,3938.4,15144.1,22744.4,14198.8,9360.4,6219.5,16554.9,16622.6,13488.7,13740.1,5481.3,11074.1,21412.3,23009.7,23098.3,7874.1,3966.5,11364.2,20621.6
SSTFMWVLHCHKNG,44865.8,42298.8,38909.9,42717.9,25716.2,39557.0,40608.0,43890.1,30252.2,15498.1,40268.4,35903.4,41637.4,43457.6,35753.7,34556.2,41097.6,32160.7,34584.8,43866.0,41238.7,26830.0,28742.7,42919.2,39493.0,42250.6,34514.1,37210.7,37497.0,36667.9,38414.4,31386.2,35609.3,27734.9,44761.2,32154.9,36404.8,21535.6,33460.4,34847.3,31448.6,33166.9,39703.3,37050.6,33174.4,36349.1,35093.6,40923.2,30340.5,38489.6,39269.6,43277.0,40119.8,34672.8,37568.4,38813.7,42587.8,45513.4,39465.4,36470.4,36334.7,34517.7,36501.0,36759.8,39608.8,37140.1,28997.9,39201.9,34168.3,40416.3,31511.1,23702.6,24459.9,37410.4,40441.5,39727.9,43281.4,42200.1,41301.8,41022.1,38716.0,38079.6,41414.7,44625.9,44299.1,42349.6,44097.4,46688.8,40452.1,40547.5,40968.6,41460.3,41161.2,41760.6,36773.2,37735.2,43469.9,41742.1,42940.2,39345.1,40750.3,38906.1,43829.8
NDYRIIHVH,29750.7,7903.2,39663.4,26269.3,26520.4,17741.9,6530.7,23519.5,16783.4,24907.1,38796.0,34045.0,37236.8,32764.2,33910.4,31876.9,36690.2,26640.4,34771.6,37988.3,38108.1,22606.4,22956.5,33713.4,23577.7,33039.6,28233.4,29657.2,20610.8,22948.6,22252.9,18901.8,9758.7,12195.3,32876.3,13544.5,27628.0,11451.6,26564.7,17732.1,22462.4,12031.0,28807.3,28633.4,20666.2,28277.9,23257.5,25003.9,11971.8,23476.4,17776.0,38533.8,33476.2,984.9,22771.0,23104.9,15921.4,31338.4,27375.9,25519.2,21732.5,10818.2,26628.9,18322.9,12300.8,17540.1,22481.6,25575.0,31230.6,28830.2,23186.7,22908.0,22448.4,12387.8,35440.0,29550.4,22569.4,22414.0,34456.6,27987.0,33094.3,26622.5,21300.3,29112.5,17602.3,17604.6,26471.7,14319.8,26587.4,24828.2,23827.5,38974.0,37958.0,27690.5,24379.9,14117.3,26238.3,16518.9,32692.1,23436.5,29298.5,21037.6,9746.2
NEVLWRNRILEIIN,43661.7,42466.3,41698.3,43109.4,35534.9,38811.3,33366.6,43465.4,29895.3,32658.2,42471.5,39083.2,42304.8,41013.3,40566.9,39243.5,41674.1,38714.2,38754.7,43677.4,42242.6,33804.1,37618.9,43068.1,39876.2,42639.4,35534.9,38686.6,39092.5,37827.9,41863.4,39871.7,40872.5,34461.3,46553.1,37443.4,36962.9,32868.0,42030.3,38619.2,31571.3,33145.8,40389.2,35203.0,32552.2,37443.5,36114.8,40695.8,33422.3,38902.1,40563.2,44825.5,44491.1,21154.4,37869.9,38601.7,41773.6,44984.0,39218.1,35042.4,28128.9,27476.4,37155.8,31523.1,30333.4,30040.7,30515.9,34905.5,36266.3,41366.6,41046.5,34295.9,31764.1,30961.9,40911.5,41612.8,43358.8,42536.8,41822.6,42428.9,45258.0,47372.9,39653.2,44025.0,39708.5,41722.2,44182.7,47429.4,39704.9,40816.4,40724.4,43114.5,44056.6,43817.1,40210.9,40351.9,43651.0,41203.5,43747.3,40571.0,43173.7,40487.6,43099.8
TDPNYHTHFSVT,44066.4,36674.1,32271.2,38109.4,21052.2,36108.5,36135.4,41980.7,30404.2,22295.5,41215.5,34035.9,41445.4,39809.0,36623.4,37166.7,39817.9,34899.7,34782.4,42846.6,41618.2,35189.1,37225.1,41790.5,39337.7,41729.1,35779.5,38304.8,38869.4,36969.1,41858.0,39576.0,39532.8,35183.1,45296.5,40505.4,36365.8,35874.3,36030.8,39184.3,30853.7,32919.7,38420.4,33262.2,27467.2,33073.5,34039.6,41103.6,32110.1,38958.8,39464.1,42709.4,44457.7,34432.4,33556.3,33590.2,40714.1,44637.0,38843.7,36048.2,28612.5,26040.4,30697.6,36461.0,39668.4,35404.3,29492.7,33964.7,32276.1,43412.5,29774.3,32512.1,32916.2,31862.0,39917.3,39059.0,42346.8,41351.6,40547.8,39095.9,33910.1,45607.0,41771.1,43517.8,41057.9,41285.8,43851.4,46378.9,40046.9,39228.6,39703.3,41008.7,41230.3,41780.3,38014.6,40518.1,43804.4,41473.9,42605.4,37678.0,39528.8,37422.5,39159.1
HQWFQCVAMQSY,27487.0,30700.5,37273.9,35376.0,29078.2,27724.0,20597.0,30480.4,17310.7,16021.7,37523.0,33961.1,40969.4,19968.6,34875.0,31825.6,37331.1,31175.8,29412.7,41535.5,40642.4,12424.8,23533.8,35308.2,33245.6,39775.3,22472.1,17719.2,19849.4,24844.8,7505.9,31364.7,4307.2,16799.1,27251.2,27379.8,25184.4,23448.3,35342.1,32138.7,29292.0,19190.3,30915.6,30562.0,31545.7,35961.0,30709.6,30188.3,499.8,14875.3,5731.1,34921.0,31566.9,5655.9,8402.1,1542.5,14847.3,32801.6,25414.0,23775.9,29195.9,18812.6,35386.2,4887.0,7298.3,19041.1,13830.1,30353.9,34462.2,23459.1,27710.9,18963.8,23748.8,14531.7,35047.3,36914.8,25988.6,29764.9,29475.3,39036.2,39794.4,29819.9,26826.0,32217.7,10013.9,26916.3,34912.0,37665.9,24758.7,17295.9,28531.8,24764.5,16139.5,30351.5,20914.8,27508.2,32628.3,28530.1,26332.1,16659.9,32297.3,30264.2,32507.1
IWQVYIQCGTEM,41061.7,37961.5,25302.7,38762.8,17765.9,10623.3,31859.8,31245.1,31023.4,25442.0,30489.9,35963.1,37256.7,35425.5,37360.4,37226.3,37086.9,36788.1,35657.8,41540.2,35619.8,32962.4,37114.7,20325.1,20101.6,32000.1,32835.6,34734.3,34929.6,33865.9,29462.0,35562.1,26925.2,30392.0,43469.1,35884.5,34001.2,36984.6,33796.5,39019.3,29735.5,32695.2,33217.4,16898.4,22006.2,33778.4,34261.2,38876.7,12971.4,29024.0,24142.0,39208.5,36860.8,23883.6,30132.9,22085.8,22174.9,36579.1,34827.9,33265.2,26867.1,27697.8,30006.4,26056.1,35488.9,31119.9,22043.6,33850.8,22722.1,23271.5,39360.0,25160.2,26966.6,22818.3,34267.7,30985.1,30357.2,27903.9,30255.5,30990.4,28298.9,30412.8,38456.8,39932.5,35373.0,34095.3,39746.8,44862.0,21759.5,21611.7,35509.8,40609.4,40356.5,36262.1,30415.4,38309.5,40677.9,32454.1,34949.9,8240.3,38245.3,27499.7,38969.8
WNPMPYADKDN,43793.1,41967.1,40595.5,38058.9,26509.5,29836.7,35333.9,43560.0,31045.3,24804.1,41242.6,39742.3,40936.8,42471.1,39179.9,38693.2,41403.3,37812.8,38097.8,43532.7,40502.2,34092.6,36870.7,42033.7,29201.6,41949.2,32912.4,35608.2,36362.6,32689.5,35304.0,39432.3,27060.9,33299.5,46315.2,38181.4,35417.9,33307.1,39797.7,37844.6,31558.8,32729.9,38648.1,34952.4,31312.2,34866.6,35333.3,39596.8,24432.7,37399.1,34958.8,36498.7,43456.6,37726.4,30345.3,29272.5,40734.0,44974.5,37794.7,36123.6,34346.3,31343.8,35849.8,37541.7,43371.3,38712.9,27773.2,38014.3,34428.3,43181.5,43353.8,25804.8,30897.2,38150.3,39755.9,39856.5,42141.5,40976.5,40968.5,41109.5,21640.6,36394.9,38594.2,43406.0,41654.3,40884.6,43121.2,46926.9,33685.5,38138.6,39382.6,36892.5,38091.4,41887.0,30276.8,30591.1,39255.5,39942.9,42992.5,34800.4,35997.4,39474.3,42450.7
DYLTMYNLAGHYMF,41216.3,39319.5,37516.7,38946.9,24901.2,18489.0,36248.8,41018.7,29045.8,30236.2,40621.7,40522.9,42015.7,40969.5,38931.0,38020.7,41315.7,36615.3,36461.1,43425.3,42086.8,32686.2,34084.9,4938.4,5634.0,22583.4,31068.1,33487.8,34530.0,34429.4,14893.3,38319.0,33635.5,29912.6,38086.4,18012.7,34384.5,28858.3,36132.1,31816.5,29143.5,30528.5,37580.3,35468.6,23500.5,33626.4,31227.5,37015.5,30141.8,37652.6,36980.8,41228.8,37043.9,11156.5,34073.5,32101.7,21944.1,35367.7,32575.7,31466.1,33729.3,30785.8,33940.0,34077.9,32036.2,36565.2,27349.4,37480.3,30117.8,34138.7,38034.4,28652.9,30980.1,31970.7,38401.2,39857.3,39077.2,35957.9,36971.2,40692.9,38297.0,42915.1,38921.3,38533.3,41037.2,38777.0,40785.5,42972.7,39144.8,37696.9,39104.8,37455.3,37600.0,40946.1,32211.7,37757.2,39846.1,37276.1,31433.1,26746.3,39308.8,32511.1,41919.2
CPGSCSNVEWFTSA,43052.5,41000.2,37276.6,40353.2,30839.2,33841.4,39836.6,37541.9,26747.9,19091.0,40729.1,37009.5,41693.5,38949.3,36880.0,36074.5,39750.1,34468.0,36268.0,41459.3,41595.7,31913.6,36962.6,42242.1,38673.8,41821.9,35358.2,38802.6,38705.1,38263.5,41344.8,39993.5,35732.9,34531.4,45588.8,40921.0,36880.4,38059.4,36557.1,39666.9,29659.7,32871.4,33048.7,13785.4,11061.9,25479.4,26941.0,40189.6,33952.7,38843.5,39758.0,44166.2,44126.5,32801.7,37107.9,36678.8,18676.0,36214.4,38034.0,34374.1,35037.7,33855.5,18579.0,37470.3,40714.5,26607.2,29370.0,38876.6,30730.1,26538.1,5463.4,33028.9,32508.0,28632.8,40303.0,39843.1,42588.1,41675.9,41191.6,40928.5,43912.4,47001.9,42041.0,44321.2,43281.3,37725.2,43718.0,46322.0,41703.2,40945.5,40231.8,42291.5,41186.2,38908.2,37563.7,41088.1,40616.9,39180.0,39582.2,37770.9,41653.1,32565.9,43315.5
TWLEAGSCNKFWCHY,43256.5,39229.2,31846.7,41249.1,22922.0,35472.1,37847.5,39274.3,16725.9,27082.6,43111.9,39658.3,42651.4,38557.6,39526.6,38566.8,40368.5,37618.1,37232.6,43056.4,42567.0,19631.5,29886.2,16257.7,27592.6,35667.9,34360.5,34792.0,36311.3,35826.1,8614.4,36054.8,17277.1,27552.0,41202.4,32120.8,35727.7,35292.4,40278.2,37839.7,31664.9,26671.4,39115.9,37688.0,34587.6,36326.4,36058.9,41403.8,16078.0,31715.7,31813.5,45128.2,40035.7,22478.7,35077.4,31707.1,28745.7,39056.4,39072.2,36471.8,36018.2,34942.6,37269.5,32201.6,29156.5,34952.9,29944.8,40025.6,34986.7,32836.4,38958.8,30948.0,30916.8,30316.3,40416.7,40818.4,32363.8,34988.8,37628.4,41676.8,43469.2,44015.7,41286.8,45040.6,42839.7,37323.5,42974.6,44799.3,39990.6,40266.6,39825.9,40864.0,41757.8,29762.6,32694.0,26427.3,41433.5,31461.3,37996.6,20136.4,41865.3,38221.3,43083.3
EEATDSRNYMRRL,43309.9,39975.1,36980.8,40770.0,26845.4,31663.4,18520.3,40400.6,32317.1,28377.1,38169.4,35509.3,39890.3,41532.5,38309.8,38115.8,40846.9,36438.1,37093.9,42930.5,39579.2,34478.8,37658.8,40705.5,38297.7,41800.1,33067.9,33014.0,32985.0,34258.1,41326.6,38227.3,39134.9,32951.4,42209.4,41582.2,33913.4,37663.4,33216.3,38646.4,30554.2,33276.9,33973.5,30058.9,25584.4,33144.6,32602.9,38320.0,33672.5,39174.1,40919.7,42064.5,40530.9,28407.8,33487.5,32993.4,42157.0,44495.1,34922.9,31182.7,21642.8,18956.3,29151.9,23436.5,17335.5,28156.7,30184.3,19678.7,31271.4,39334.2,41414.2,31418.8,31800.9,16677.8,38726.9,35946.1,37806.4,36625.3,40075.2,36961.7,37460.0,40782.8,38173.0,42921.7,31447.5,38095.7,42302.2,46969.4,34483.4,35123.9,35823.2,43203.7,44314.5,30994.0,34785.4,35043.5,42765.9,39533.0,35406.3,35756.6,39472.2,30912.6,39842.7
MSGDACND,39531.0,39404.3,42689.8,36631.0,24512.5,31873.8,22499.5,35348.8,27067.7,27959.4,36059.9,36380.6,40222.8,39885.0,35657.4,36346.6,37150.1,33696.8,35618.9,39815.2,37296.2,30732.1,30971.3,38186.1,33521.8,38861.9,32692.8,37045.0,37182.4,35007.3,35006.2,30627.5,21765.5,28733.2,38896.4,31030.8,34166.9,25154.9,37154.9,33131.6,26235.8,28117.1,32986.3,35470.5,26935.0,28876.1,27638.4,38167.8,21090.3,31994.7,32821.9,38174.9,26573.1,31407.1,35055.4,35544.4,35526.5,41595.8,36912.0,35058.5,32903.2,31917.8,28423.1,34514.3,39786.8,33805.7,23372.3,35819.7,25962.6,35199.4,35434.9,20873.3,10865.1,30611.6,26672.0,16901.8,32916.3,33854.9,34604.6,15853.9,22614.9,36981.8,36604.8,42344.9,37295.1,38385.0,39678.6,45767.9,36136.8,38201.6,35585.8,29834.5,26730.9,29939.4,17543.7,24477.3,36685.4,33534.9,40217.3,34840.3,25639.9,33476.0,38813.7
""".strip()

def run_and_check(n_jobs=0, delete=True, additional_args=[]):
    models_dir = tempfile.mkdtemp(prefix="mhcflurry-test-models")
    hyperparameters_filename = os.path.join(
        models_dir, "hyperparameters.yaml")
    with open(hyperparameters_filename, "w") as fd:
        json.dump(HYPERPARAMETERS_LIST, fd)

    pretrain_data_filename = os.path.join(
        models_dir, "pretrain_data.csv")
    with open(pretrain_data_filename, "w") as fd:
        fd.write(PRETRAIN_DATA)
        fd.write("\n")

    data_df = pandas.read_csv(
        get_path("data_curated", "curated_training_data.affinity.csv.bz2"))
    selected_data_df = data_df.sample(frac=0.1)
    selected_data_df.to_csv(
        os.path.join(models_dir, "_train_data.csv"), index=False)

    args = [
        "mhcflurry-class1-train-pan-allele-models",
        "--data", os.path.join(models_dir, "_train_data.csv"),
        "--allele-sequences", get_path("allele_sequences", "allele_sequences.csv"),
        "--pretrain-data", pretrain_data_filename,
        "--hyperparameters", hyperparameters_filename,
        "--out-models-dir", models_dir,
        "--num-jobs", str(n_jobs),
        "--num-folds", "2",
        "--verbosity", "1",
    ] + additional_args
    print("Running with args: %s" % args)
    subprocess.check_call(args)

    # Run model selection
    models_dir_selected = tempfile.mkdtemp(
        prefix="mhcflurry-test-models-selected")
    args = [
        "mhcflurry-class1-select-pan-allele-models",
        "--data", os.path.join(models_dir, "train_data.csv.bz2"),
        "--models-dir", models_dir,
        "--out-models-dir", models_dir_selected,
        "--max-models", "1",
        "--num-jobs", str(n_jobs),
    ] + additional_args
    print("Running with args: %s" % args)
    subprocess.check_call(args)

    result = Class1AffinityPredictor.load(
        models_dir_selected, optimization_level=0)
    assert_equal(len(result.neural_networks), 2)
    predictions = result.predict(peptides=["SLYNTVATL"],
        alleles=["HLA-A*02:01"])
    assert_equal(predictions.shape, (1,))
    assert_array_less(predictions, 1000)

    if delete:
        print("Deleting: %s" % models_dir)
        shutil.rmtree(models_dir)
        shutil.rmtree(models_dir_selected)


def test_run_parallel():
    run_and_check(n_jobs=1)
    run_and_check(n_jobs=2)


def test_run_serial():
    run_and_check(n_jobs=0)


def test_run_cluster_parallelism():
    run_and_check(n_jobs=0, additional_args=[
        '--cluster-parallelism',
        '--cluster-results-workdir', '/tmp/'
    ])


if __name__ == "__main__":
    # run_and_check(n_jobs=0, delete=False)
    test_run_cluster_parallelism()
