# spec-decoding

Toy repo to play with **speculative decoding** vs plain greedy decoding.  
Draft = GPT-2 small, Verifier = GPT-2 medium. Just enough to show speedups and acceptance rates.

---

## what this is
- speculative decoding: small model proposes a few tokens, big model checks them in one go  
- compare runtime vs greedy baseline  
- logs acceptance rate + a rough quality proxy (avg log-likelihood)  
- code is simple and hacky, but readable  

---

## install

```bash
git clone https://github.com/vincentcounathe/spec-decoding
cd spec-decoding
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
