# ImmuneFold

## Introduction
**ImmuneFold** is an advanced approach for predicting immune protein structures using transfer learning, adapted from state-of-the-art protein structure prediction frameworks. ImmuneFold is specifically tailored for accurate modeling of immune proteins, including T-cell receptors (TCRs), antibodies, nanobodies, and their complexes with target antigens. By providing precise predictions of immune protein-antigen pairings, ImmuneFold offers valuable insights into protein interaction mechanisms, thereby supporting applications such as vaccine development and immune response analysis.

## Installation
To install ImmuneFold, the recommended method is to create a conda environment and install the required dependencies:

```bash
git clone git@github.com:CarbonMatrixLab/immunefold.git 
conda env create -f environment.yml
pip install fair-esm
```

Additionally, to compute the mutation effects on TCR-pMHC interactions, please install the [PyRosetta package](https://www.pyrosetta.org/downloads).

## Model Weights
1. Download the [**ImmuneFold-TCR**](https://immunefold.s3.amazonaws.com/immunefold-tcr.ckpt) and [**ImmuneFold-Ab**](https://immunefold.s3.amazonaws.com/immunefold-ab.ckpt) here and place them in the `./params` directory.
2. Download the **ESM2 model** weights from [this link](https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt) and the **contact regressor** weights from [here](https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t33_650M_UR50D-contact-regression.pt). Save these files in the `./params` directory.

## Usage

### TCR-pMHC Structure Prediction
To predict the structure of TCR-pMHC complexes, provide the TCR, peptide, and MHC sequences in a FASTA file, `TCR_B_A_P_M.fasta`, where B, A, P and M denote the beta, alpha, peptide, MHC chain ids, respectively. The format is as follows `Beta_seq:Alpha_seq:Peptide_seq:MHC_seq`:

```
>TCR_B_A_P_M
VSQHPSWVICKSGTSVKIECRSLDFQATTMFWYRQFPKQSLMLMATSNEGSKATYEQGVEKDKFLINHASLTLSTLTVTSAHPEDSSFYICSVSRDRNTGELFFGEGSRLTVL:VEQDPGPFNVPEGATVAFNCTYSNSASQSFFWYRQDCRKEPKLLMSVYSSGNEDGRFTAQLNRASQYISLLIRDSKLSDSATYLCVVNEEDALIFGKGTTLSVSS:YLQPRTFLL:GSHSMRYFFTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASQRMEPRAPWIEQEGPEYWDGETRKVKAHSQTHRVDLGTLRGYYNQSEAGSHTVQRMYGCDVGSDWRFLRGYHQYAYDGKDYIALKEDLRSWTAADMAAQTTKHKWEAAHVAEQLRAYLEGTCVEWLRRYLENGKETLQ
```

To run the ImmuneFold-TCR structure prediction model, use the following command:

```bash
python inference.py --config-name=TCR_structure_prediction
```

### Zero-Shot Binding Affinity Prediction for TCR-pMHC
To predict binding affinity using zero-shot learning, provide the whole TCR-pMHC sequences in a FASTA file, following the same format as described for structure prediction:

```bash
python inference.py --config-name=TCR_structure_prediction
python predict_energy.py --pdb_dir /path/to/pdb/dir --name_idx /path/to/name_idx.idx --output_file /path/to/energy.csv --mode interface
```

### Unbound Antibody or Nanobody Structure Prediction
For antibody or nanobody structure prediction, provide the sequences in FASTA files, `antibody_H_L.fasta` or `nanobody_H.fasta`, where H and L represent the heavy and light chain ids, respectively. The formats are as follows:

**Antibody:**
```
>antibody_H_L
VSQHPSWVICKSGTSVKIECRSLDFQATTMFWYRQFPKQSLMLMATSNEGSKATYEQGVEKDKFLINHASLTLSTLTVTSAHPEDSSFYICSVSRDRNTGELFFGEGSRLTVL:VEQDPGPFNVPEGATVAFNCTYSNSASQSFFWYRQDCRKEPKLLMSVYSSGNEDGRFTAQLNRASQYISLLIRDSKLSDSATYLCVVNEEDALIFGKGTTLSVSS
```

**Nanobody:**
```
>nanobody_H
VSQHPSWVICKSGTSVKIECRSLDFQATTMFWYRQFPKQSLMLMATSNEGSKATYEQGVEKDKFLINHASLTLSTLTVTSAHPEDSSFYICSVSRDRNTGELFFGEGSRLTVL
```

To run the model:

```bash
python inference.py --config-name=antibody_structure_prediction
```

or for nanobodies:

```bash
python inference.py --config-name=nanobody_structure_prediction
```

### Bound Antibody or Nanobody Structure Prediction with Target Antigen
For predicting antibody or nanobody structures bound to a target antigen, provide the antigen structure as a PDB file `antigen.pdb` along with the antibody or nanobody sequences, and run the following command:

```bash
python inference.py --config-name=antibody_antigen_structure_prediction
```


