import casadi
import os

# def addFuncsasAttributes(obj,)

class DigitCasFunc():
    """Load generated CasADi functions into a nested attribute hierarchy."""

    def __init__(self,pathtoGeneratedfolder='Generated/',usesharedobjects=True):
        """Initialize and load generated functions.

        :param pathtoGeneratedfolder: Path relative to this module.
        :type pathtoGeneratedfolder: str
        :param usesharedobjects: Load compiled libraries when true.
        :type usesharedobjects: bool
        """
        thisFileDir = os.path.dirname(os.path.realpath(__file__))
        self.GenFolder = thisFileDir+'/'+pathtoGeneratedfolder
        if usesharedobjects:
            self.PS = '.so'
            os.system('make -C '+self.GenFolder)
        else:
            self.PS = '_gen.c'
        
        self.addFuncaasAttributesRecursively(self.GenFolder)

        #now adding each function as methods
        
    
    def addFuncaasAttributesRecursively(self,dir,subAttr=''):
        """Recursively expose generated functions as object attributes.

        :param dir: Directory to scan; it must include a trailing separator.
        :type dir: str
        :param subAttr: Dotted attribute prefix for the current directory.
        :type subAttr: str
        """
        print(dir)
        for ldir in os.listdir(dir):
            if os.path.isdir(dir+ldir):
                #call recursively
                exec("self"+subAttr+'.'+ldir+"=DigitInternalCasFunc('From Directory"+dir+"')")#execute arbitary code
                self.addFuncaasAttributesRecursively(dir+ldir+'/',subAttr+'.'+ldir)
            else:
                #add add attributes
                if ldir.endswith(self.PS):
                    funcName = ldir[:-len(self.PS)] #remove self.PS for end of file name
                    if self.PS.endswith('.c'):
                        print("Generating "+funcName)
                        exec("C = casadi.Importer(dir+'/'+ldir,'clang')")
                        exec("setattr(self"+subAttr+",'"+funcName+"',casadi.external(funcName,C))")
                    else:
                        exec("setattr(self"+subAttr+",'"+funcName+"',casadi.external(funcName,dir+'/'+ldir))")


class DigitInternalCasFunc():
    """Namespace populated dynamically with generated CasADi functions."""

    def __init__(self,info='NoneYet'):
        """Create a generated-function namespace.

        :param info: Human-readable namespace description.
        :type info: str
        """
        self.info=info 
