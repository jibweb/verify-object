Last update: 15.11.22

## Status
- [ ] Initial implementation - *by mid January*
- [ ] Prepare evaluation - *by mid January*
- [ ] Tuning and Ablation - *January/February*
- [ ] Writing - *February*

### Status: Initial Implementation
- [x] Basic inverse rendering loop with mask-based loss
- [x] Basic version of collision-based loss
- [ ] Basic version of contour-based loss - *by mid December*
- [ ] Extension to multiple objects per frame - *by mid January*
  - [x] with mask-based loss - *by mid November*
  - [ ] with collision-based loss
  - [ ] with contour-based loss

### Status: Prepare evaluation
- [x] Custom TraceBot dataset
- [ ] [BOP datasets](https://bop.felk.cvut.cz/datasets/) - *by mid January*
  - [ ] HOPE
  - [ ] T-LESS

### Status: Tuning and Ablation (*actual points to be determined*)
For complete initial implementation...
- [ ] Tune rendering/optimization loop (representation, LR, optimizer)
- [ ] Tune losses (weighting, hyperparameters, computation of individual loss)
- [ ] Ablation over initial pose and mask error

Goal: Find settings for "best" results we can get, motivate parameter choices and thoroughly evaluate method (for thesis).