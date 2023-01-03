# A Code-T5 based Solidity Lauguage Model for Smart Contract Code Generation

## Published model at HuggingFace
- See https://huggingface.co/hululuzhu/solidity-t5
- Hello World example to utilize the trained model
  - A hello world example to use this model, notice the input `text` includes
    - Header solidity version like `pragma solidity ^0.5.7`
    - Ancestor class/library info, e.g. public functions and constants from `ParentA`
    - Contract/Library/Interface declaration header, e.g. `HelloWorld` ended with `{`
    ```python
    # !pip install transformers -q

    from transformers import AutoTokenizer, T5ForConditionalGeneration

    DEVICE = 'cuda'  # fallback to cpu if you do not have cuda
    tokenizer = AutoTokenizer.from_pretrained("hululuzhu/solidity-t5")
    model = T5ForConditionalGeneration.from_pretrained("hululuzhu/solidity-t5").to(DEVICE)

    text = """pragma solidity ^0.5.7;
    // Context: ParentA | Functions: helloA helloB | Constants: constantA 
    contract HelloWorld is ParentA {"""
    input_ids = tokenizer(text, return_tensors="pt", truncation=True).input_ids.to(DEVICE)

    # Need to tune beam/topk/topp params to get good outcome
    generated_ids = model.generate(input_ids, max_length=256, num_beams=5, top_p=0.95, top_k=50)
    print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))

    # Expect outcome
    """
    string public constant name = "Hello World";
    ...
    uint256 public constant override returns (uint256) {
    return initialSupply;
    }
    function initialSupply() public view returns (uint256) {
    ...
    """
    ```

## Background
- Base T5 code model: https://huggingface.co/Salesforce/codet5-large
- Source data: https://huggingface.co/datasets/mwritescode/slither-audited-smart-contracts
  - Processing steps: Clean, contract-level segmentation sepration, split in and out
  - After processing input sample

    ```
    pragma solidity 0.5.7;
    // Context: PauserRole | Functions: isPauser addPauser renouncePauser | Constants: 
    contract Pausable is PauserRole {
    ```

  - After processing output sample (**notice indentation is bad, this is intentional to reduce token size**)

    ```
    event Paused(address account);
    event Unpaused(address account);
    bool private _pausableActive;
    bool private _paused;
    constructor () internal {
    _paused = false;
    }
    function paused() public view returns (bool) {
    return _paused;
    }
    modifier whenNotPaused() {
    require(!_paused);
    _;
    }
    modifier whenPaused() {
    require(_paused);
    _;
    }
    function pause() public onlyPauser whenNotPaused whenPausableActive {
    _paused = true;
    emit Paused(msg.sender);
    }
    function unpause() public onlyPauser whenPaused whenPausableActive {
    _paused = false;
    emit Unpaused(msg.sender);
    }
    function _setPausableActive(bool _active) internal {
    _pausableActive = _active;
    }
    modifier whenPausableActive() {
    require(_pausableActive);
    _;
    }
    }
    ```
- Source training code: See the [end to end notebook](https://github.com/hululuzhu/solidity-t5/blob/main/code/Solidity_T5_Data_Processing_and_Training.ipynb) at code dir here

## Future TODO
- The model is significantly under-trained because of lack of GPU budget, need 10x colab resources (~$100 for full train)
- This is quite limited on how the model is used, potentially we could switch to GPT2 decoder-only to compare, but CodeT5 has its strong code optimization
- Need more classifiers (T5 or BERT alike) to detect potential defects.
  - This is the intention of the project, however I found it quite challenging to find the labeled data that points the exact line of code that has defect
  - Technically, 

## Some thoughts (01/2023)
- As I tested a few examples using this significantly-under-trained model and compared with chatGPT, it seems they perform similarly for code completion. That shows the great potential for this model to surpass chatGPT if we have 30x training budget (and reliable training pipeline)
  - 30x budget? I trained 3 hours which only finished 10% of 1 epoch, I expect 3 epoches are reasonable size for finetune based on my personal experience
- The data is specially tuned for codeT5, thus it has the limitation of
  - Split to input and output
  - In/Out token size limit is 512
- Ideally we could change the transformed data, so it could be trained in BERT style to compare with T5-based classifiers, or fill-missing-splan style to add context before and after to let model output middle code
- Training more classifiers (T5 or BERT alike) to detect potential defects is the intention of the project, however I found it quite challenging to find the labeled data that points the exact line of code that has defect
  - The part I divide code into segment and their ancestor info could be useful, but need more time to evaluate
  - Technically, 
- It is also hard to tell if my aggressive approach to remove all comments (thus rely on meaning of code only) is a good approach
