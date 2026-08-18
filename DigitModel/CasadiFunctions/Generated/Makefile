ALLSRC = $(wildcard **/*_gen.c **/*/*_gen.c) #Search for source files in 2 depths
ALLOBJS = $(patsubst %_gen.c,%.so,$(ALLSRC)) #Make names of desired objects for each source fiel
CC := gcc
CFLAGS := -fPIC -shared 

#all:
#	@echo $(ALLOBJS) #for debugging

default : $(ALLOBJS)

$(ALLOBJS) : $(ALLSRC)
	$(CC) $(CFLAGS) $(subst .so,_gen.c,$@) -o $@
#	@echo $@ $(subst .so,_gen.c,$@) #subst is a substitute string function

.PHONY : clean
clean :
	-rm $(ALLOBJS)
