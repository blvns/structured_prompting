# Codebase for "Prompting Language Models for Linguistic Structure"

This codebase contains the code used to implement the structured prompting method proposed in our ACL 2023 paper. 

## How to Run
This code requires the following libraries:
* [Python 3](https://www.python.org/)
* [Pytorch](https://pytorch.org/)
* [Huggingface Transformers](https://huggingface.co/docs/transformers/index)
* [Huggingface Datasets](https://huggingface.co/datasets)
* [Huggingface Accelerate](https://huggingface.co/docs/accelerate/index)
* [scikit-learn](https://scikit-learn.org/stable/)
* [Numpy](https://numpy.org/)
* [tqdm](https://tqdm.github.io/)

To run, you will also need to fill in the indices and results filepaths (lines 29 and 31 in the *structured_prompting* script) as appropriate for your environment. The *indices filepath* should point to the directory with the indices provided in this repository (note that new index mappings will need to be created for new benchmarks or UD treebanks); the *results filepath* should point to where the prompting results should be saved. 

If you want to perform structured prompting on UD datasets for POS tagging, you will need to fill the relevant information into lines 187-189 of the *utils* script.

You can also use OpenAI model with the OpenAI API, but you need to uncomment the import statements in the *structured_prompting* script and install the respective library. You also need to add your personal api_key to the script at line 22.


## Citation
If you use this work, please cite the corresponding [paper](https://blvns.github.io/papers/blevins2023prompting.pdf):
```
@inproceedings{
  blevins2023prompting,
  title={Prompting Language Models for Linguistic Structure},
  author={Terra Blevins and Hila Gonen and Luke Zettlemoyer},
  booktitle={Proceedings of the 61st Association for Computational Linguistics},
  year={2023},
  url={https://blvns.github.io/papers/blevins2023prompting.pdf}
}
```
