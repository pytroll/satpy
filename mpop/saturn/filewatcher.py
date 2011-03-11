#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2010.

# SMHI,
# Folkborgsvägen 1,
# Norrköping, 
# Sweden

# Author(s):
 
#   Martin Raspaud <martin.raspaud@smhi.se>

# This file is part of mpop.

# mpop is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.

# mpop is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with
# mpop.  If not, see <http://www.gnu.org/licenses/>.

"""Watch files coming in a given directory.
"""
import datetime
import glob
import os
import signal
import sys
from Queue import Empty

import time
from multiprocessing import Process

from mpop.saturn import LOG


ZERO = datetime.timedelta(seconds=0)

class FileWatcher(Process):
    """Looks for new files, and queues them.
    """

    def __init__(self, filename_template, file_queue, frequency):
        """Looks for new files arriving at the given *frequency*, and queues
        them.
        """
        Process.__init__(self)
        self.queue = file_queue
        self.template = filename_template
        self.frequency = datetime.timedelta(minutes=frequency)
        self.running = True

    def run(self):
        """Run the file watcher.
        """

        def stop(*args):
            """Stops a running process.
            """
            del args
            self.running = False
            
        signal.signal(signal.SIGTERM, stop)

        filelist = set()
        
        while self.running:
            if isinstance(self.template, (list, tuple)):
                new_filelist = []
                for template in self.template:
                    new_filelist += glob.glob(template)
                new_filelist = set(new_filelist)
            else:
                new_filelist = set(glob.glob(self.template))
            files_to_process = list(new_filelist - filelist)
            filelist = new_filelist

            files_dict = {}
            for fil in files_to_process:
                files_dict[fil] = os.path.getmtime(fil)

            files_to_process.sort(lambda x, y: cmp(files_dict[x],
                                                  files_dict[y]))

            if len(files_to_process) != 0:
                sleep_time = 8
                times = []
                for i in files_to_process:
                    LOG.debug("queueing %s..."%i)
                    self.queue.put(i)
                    times.append(os.stat(i).st_ctime)
                times.sort()

                since_creation = datetime.timedelta(seconds=time.time() -
                                                    times[-1])
                if(self.frequency > since_creation):
                    to_wait = self.frequency - since_creation

                    LOG.info("Waiting at least "+str(to_wait)+" for next file")
                    time.sleep(to_wait.seconds +
                               to_wait.microseconds / 1000000.0)
            else:
                LOG.info("no new file has come, waiting %s secs"
                         %str(sleep_time))
                time.sleep(sleep_time)
                if sleep_time < 60:
                    sleep_time *= 2


class FileProcessor(Process):
    """Execute *fun* on filenames provided by from *file_queue*. If *refresh*
    is a positive number, run *fun* every given number of seconds with None as
    argument.
    """
    def __init__(self, file_queue, fun, refresh=None):
        Process.__init__(self)
        self.queue = file_queue
        self.fun = fun
        self.running = True
        self.refresh = refresh
        
    def run(self):
        """Execute the given function on files from the file queue.
        """
        def stop(*args):
            """Stops a running process.
            """
            del args
            self.running = False
            sys.exit()
        signal.signal(signal.SIGTERM, stop)

        while self.running:
            try:
                filename = self.queue.get(block=True, timeout=self.refresh)
                LOG.debug("processing %s"%filename)
            except Empty:
                filename = None
                LOG.debug("refreshing.")
            try:
                self.fun(filename)
            except:
                LOG.exception("Something wrong happened in %s for %s. Skipping."
                              %(str(self.fun), filename))

    def stop(self):
        """Stops a running process.
        """
        self.running = False



