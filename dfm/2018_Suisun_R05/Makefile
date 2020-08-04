DFLOWFM=dflowfm
MPIEXEC=mpiexec.mpich
NP=4

MDU=FlowFM.mdu

.PHONY: run partition

run:
	$(MPIEXEC) -n $NP $(DFLOWFM) --autostartstop $(MDU)

partition:
	$(DFLOWFM) --partition:ndomains=$(NP):icgsolver=6 $(MDU)


