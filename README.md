# undergrad_thesis
Various scripts I wrote to calculate properties of quantum spin lattice systems via exact diagonalization

This will probably not get organized beyond this README, where I'll try to list roughly what each script does. Take it with a grain of salt, I'm going back after years of not looking at this to try to write a brief synopsis. Some things might be slightly inacurrate, but probably not too bad

- textbook_method scripts are where I started when this research began as a summer intership with the professors who would become my advisors. These calculated various quantities (ground state energy, energy gaps, configuration and momentum correlation functions, (reduced) density matrix, entanglement entropy, wave function fidelity), and culminated with textbook_method_sparse, in which I implemented scipy sparse matrices 
- test and test2 were for me to test various functions and attempted implementations modularly, in separate scripts, before I added them to main code
- old_code was various attempts of mine to automate the calculation of a system's entanglement entropy. Eventually one of them worked and was quick so I used it
- binary_method was an attempt to express spin states as 0 (down) and 1 (up), and spin operators acting on these states flipping these bits back and forth. It works, but I found other methods to be more scalable, at least intuitively
- basis_class and spin_solver were an attempt to define a class, new objects, and methods basis states and how they act on a given operator (i.e. Hamiltonian). I gave up, it was too frustrating, and I figured out how to do it without all of this background structure
- chiral scripts are the core of my thesis, investigating Heisenberg interaction (nearest neighbor spin dot product, Si dot Sj) with a chiral term (particle triplet term, Si cross Sj dot Sk) added, applied to a triangular lattice
  - chiral_1d was my first pass, mostly just to get a hold of coding this new interaction
  - chiral_multi_spin was the next step, adding the triangular lattice
  - chiral_3leg was an attempt to make a 3-leg triangular lattice, which was abandoned after a bit if I remember correctly
  - chiral_tensor was roughly my final code for my thesis (I think?), roughly because I modified it slightly to run it on my research group's supercomputer cluster cleanly
