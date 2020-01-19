import pdoc

modules = ['motion', 'motion.analogs']
context = pdoc.Context()

modules = [pdoc.Module(mod, context=context) for mod in modules]
pdoc.link_inheritance(context)

modules[1].text()