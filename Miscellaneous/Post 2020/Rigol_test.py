import usbtmc

instr = usbtmc.Instrument(0x1ab1, 0x04ce)
print(instr.ask("*IDN?"))
