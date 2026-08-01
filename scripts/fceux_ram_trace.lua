-- A reference TAS trace: 2 KB of RAM per frame plus a lag flag, to a binary file.
-- FCEUX serves as the synchronisation reference — the trace tells us which frame
-- our core diverges on, and everything before it is a verified prefix.
--
-- Usage: TRACE_OUT=/path/trace.bin fceux --no-config 1 --sound 0 \
--          --loadlua scripts/fceux_ram_trace.lua --playmov movie.fm2 rom.nes
-- Format: 2049 bytes per frame — 2048 of RAM and one lag byte.
--
-- The emulator drives the loop through registerafter. A while loop with
-- frameadvance kills the Qt build around frame 300 of a movie, which is also
-- why this reference has not actually been usable so far.

local path = os.getenv("TRACE_OUT") or "/tmp/fceux_trace.bin"
local limit = tonumber(os.getenv("TRACE_FRAMES") or "0")
local out = assert(io.open(path, "wb"))

emu.speedmode("maximum")

local n = 0
emu.registerafter(function()
  if limit > 0 and n >= limit then
    out:flush()
    out:close()
    io.stderr:write("TRACE_DONE frames=" .. n .. "\n")
    os.exit(0)
  end
  out:write(memory.readbyterange(0, 2048))
  out:write(emu.lagged() and "\1" or "\0")
  n = n + 1
  if n % 2000 == 0 then
    out:flush()
    io.stderr:write("TRACE frame=" .. n .. "\n")
  end
end)

emu.registerexit(function()
  out:flush()
  out:close()
  io.stderr:write("TRACE_EXIT frames=" .. n .. "\n")
end)
