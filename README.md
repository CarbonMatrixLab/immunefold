# ImmuneFold

## Introduction
**ImmuneFold** is an advanced approach for predicting immune protein structures using transfer learning, adapted from state-of-the-art protein structure prediction frameworks. ImmuneFold is specifically tailored for accurate modeling of immune proteins, including T-cell receptors (TCRs), antibodies, nanobodies, and their complexes with target antigens. By providing precise predictions of immune protein-antigen pairings, ImmuneFold offers valuable insights into protein interaction mechanisms, thereby supporting applications such as vaccine development and immune response analysis.

## Installation
To install ImmuneFold, the recommended method is to create a conda environment and install the required dependencies:

```bash
git clone git@github.com:CarbonMatrixLab/immunefold.git 
conda env create -f environment.yml
```

Additionally, to compute the mutation effects on TCR-pMHC interactions, please install the [PyRosetta package](https://www.pyrosetta.org/downloads).

## Model Weights
1. Download the **ImmuneFold-TCR** and **ImmuneFold-Ab** [model weights](https://carbondesign.s3.amazonaws.com/params.tar) and place them in the `./params` directory.
2. Download the **ESM2 model** weights from [this link](https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt) and the **contact regressor** weights from [here](https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t33_650M_UR50D-contact-regression.pt). Save these files in the `./params` directory.

## Usage

### TCR-pMHC Structure Prediction
To predict the structure of TCR-pMHC complexes, provide the TCR, peptide, and MHC sequences in a FASTA file, `TCR.fasta`. The format is as follows:

```
>TCR_alpha
VSQHPSWVICKSGTSVKIECRSLDFQATTMFWYRQFPKQSLMLMATSNEGSKATYEQGVEKDKFLINHASLTLSTLTVTSAHPEDSSFYICSVSRDRNTGELFFGEGSRLTVL
>TCR_beta
VEQDPGPFNVPEGATVAFNCTYSNSASQSFFWYRQDCRKEPKLLMSVYSSGNEDGRFTAQLNRASQYISLLIRDSKLSDSATYLCVVNEEDALIFGKGTTLSVSS
>peptide
YLQPRTFLL
>MHC
GSHSMRYFFTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASQRMEPRAPWIEQEGPEYWDGETRKVKAHSQTHRVDLGTLRGYYNQSEAGSHTVQRMYGCDVGSDWRFLRGYHQYAYDGKDYIALKEDLRSWTAADMAAQTTKHKWEAAHVAEQLRAYLEGTCVEWLRRYLENGKETLQ
```

To run the ImmuneFold-TCR structure prediction model, use the following command:

```bash
python predict.py --config-name=TCR_structure_prediction
```

### Zero-Shot Binding Affinity Prediction for TCR-pMHC
To predict binding affinity using zero-shot learning, provide the TCR-pMHC sequences in a FASTA file, following the same format as described for structure prediction:

```bash
python predict.py --config-name=TCR_affinity_prediction
```

### Unbound Antibody or Nanobody Structure Prediction
For antibody or nanobody structure prediction, provide the sequences in FASTA files, `antibody.fasta` or `nanobody.fasta`. The formats are as follows:

**Antibody:**
```
>antibody_heavy
VSQHPSWVICKSGTSVKIECRSLDFQATTMFWYRQFPKQSLMLMATSNEGSKATYEQGVEKDKFLINHASLTLSTLTVTSAHPEDSSFYICSVSRDRNTGELFFGEGSRLTVL
>antibody_light
VEQDPGPFNVPEGATVAFNCTYSNSASQSFFWYRQDCRKEPKLLMSVYSSGNEDGRFTAQLNRASQYISLLIRDSKLSDSATYLCVVNEEDALIFGKGTTLSVSS
```

**Nanobody:**
```
>nanobody
VSQHPSWVICKSGTSVKIECRSLDFQATTMFWYRQFPKQSLMLMATSNEGSKATYEQGVEKDKFLINHASLTLSTLTVTSAHPEDSSFYICSVSRDRNTGELFFGEGSRLTVL
```

To run the model:

```bash
python predict.py --config-name=antibody_structure_prediction
```

or for nanobodies:

```bash
python predict.py --config-name=nanobody_structure_prediction
```

### Bound Antibody or Nanobody Structure Prediction with Target Antigen
For predicting antibody or nanobody structures bound to a target antigen, provide the antigen structure as a PDB file `antigen.pdb` along with the antibody or nanobody sequences, and run the following command:

```bash
python predict.py --config-name=antibody_antigen_structure_prediction
```

## Citation
If you use ImmuneFold in your research, please cite the following paper:

```
[Add your citation here]
```

