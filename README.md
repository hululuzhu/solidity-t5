# Solidity Large Lauguage Model for Code Generation

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
