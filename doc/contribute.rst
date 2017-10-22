.. _contribute:


====================
Become a contributer
====================

Why contribute
^^^^^^^^^^^^^^

As a scientist, contributing to open source software projects can increase you impact on science.
The tools you help developing can be used by many researchers, doing many discoveries.
Since the tools are openly available, lot's of double work can be avoided.
You may find that often, several researchers are working on their own to develop the same data analysis tools.

Open source software has many advantages:

1. It is better. Since the source code is open, anyone can help finding errors, improving the code and adding functionality. More about that `here <https://www.dwheeler.com/oss_fs_why.html>`_.
2. It is free. You can make the tools you want and need, not needing to wait for a company to care.
3. It is for free.

By contributing to open source projects you will gain more experience with programming and how program as a part of a community.

You can check out who is contributing to Atomap `here <https://gitlab.com/atomap/atomap/graphs/master>`_.

What to contribute
^^^^^^^^^^^^^^^^^^
Firstly, as an Atomap user you may find that you miss some functionality. Obviously, you can develop the tools you are missing, and improve existing tools. 
Contributing documentation, and doing language checks is also highly appreciated.
You can read more about contributing to open source projects `here <https://opensource.guide/how-to-contribute/>`_. 

Version control
^^^^^^^^^^^^^^^

For software development projects it is important to have a good system for keeping different versions of the software.
Atomap uses Git as version control system.
One version of Atomap is the newest released version, another is the `development version <https://gitlab.com/atomap/atomap>`_.
When new functionality is created or old functionality improved, new branches of the development version is typically made.
The branch is merged back into the master development version when they are done.

You only need to know a few basic things about git to be able to start developing.
Git is the most commonly used version control system, so there is a lot of user guides out there.
You may consult the nice Gitlab `userguide <https://docs.gitlab.com/ce/gitlab-basics/README.html>`_.
Below, you'll also find and outline of the basic first steps.
Often as with tools, the best way to learn is by using.

First steps
^^^^^^^^^^^

1. Create a user on `GitLab <https://gitlab.com/>`_ and `install <https://gist.github.com/derhuerst/1b15ff4652a867391f03>`_ git on your computer.
2. Make your own fork of Atomap `here <https://gitlab.com/atomap/atomap>`_ , from which you can do development. Click on the fork button. You need to be logged in.

.. figure:: images/misc/fork.jpg
    :scale: 75 %
    :align: center

3. Clone your fork to you computer, `as shown here <https://docs.gitlab.com/ce/gitlab-basics/command-line-commands.html#clone-your-project>`_. (Use something like ``git clone git@gitlab.com:username/atomap.git``, with your own username).
4. In the terminal, make a branch for your first development. For example a branch for fixing spelling errors, called FixTypos: ``git checkout -b FixTypos``. The -b is for creating a new branch. If the branch already exists write just ``git checkout FixTypos``. Too see what branches you have, write ``git branch``. Make sure you are in the correct branch.
5. Fix typos in a file.
6. Check what has been changed. In the terminal, in the Atomap folder: ``git status``. The file you have edited should listed in red.
7. Add your file to the stage for commit by ``git add filename``
8. Commit your file by ``git commit``. Write a short but clear commit message. Such as "Fixed typos in docstring".
9. Push your commit ``git push origin FixTypos``. This synchronizes the changes you have done to the remote version on gitlab.
10. When you are ready to add the branch to the Atomap development version, make a request for merging the commit into Atomap. See `here <https://docs.gitlab.com/ce/gitlab-basics/add-merge-request.html>`_.

When starting some new branch, remember to update your master fork by pulling from Atomap (upstream master).

1. Return to your master branch ``git checkout master``
2. Pull the newest changes from Atomap ``git pull upstream master``.
3. If upstream master is not set, set it by writing ``git remote add upstream git@gitlab.com:atomap/atomap.git``
4. Check if it is there with ``git remote -v``. Upstream should be the main Atomap project. Origin Should be you fork.

Consult the nice Gitlab `userguide <https://docs.gitlab.com/ce/gitlab-basics/README.html>`_.
Google stuff.
Or ask for help `here <https://gitlab.com/atomap/atomap/issues>`_, by adding an `issue <https://docs.gitlab.com/ce/user/project/issues/create_new_issue.html>`_.

Work on the development version
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
When testing the functionality you are developing, you need to use the development version of Atomap, and not the release version installed with pip.
Here is a tip for how to do this.

1. Make a virtual working environment for the development version. Install `virtualenvwrapper <https://virtualenvwrapper.readthedocs.io/en/latest/install.html#basic-installation>`_. Make a "workon" for Atomap development by writing in the terminal ``mkvirtualenv atomap_dev``. Got to your new "workon" by ``workon atomap_dev``.
2. In your atomap workon install the development version by going to the atomap folder in a terminal and write ``pip install -e .``

Have good habits
^^^^^^^^^^^^^^^^
Work out what you want to contribute and break it down in to manageable chunks.
When you have decided what you are going to work on - let people know using the online forums!
It may be that someone else is doing something similar and can help, it's also good to make sure that those working on related projects are pulling in the same direction.
There are 3 key points to get right when starting out as a contributor - keep work separated in manageable sections, make sure that your code style is good, and bear in mind that every new function you write will need a test and user documentation!

Learn more
^^^^^^^^^^
1. `What is git? <https://www.git-scm.com/about>`_
2. Atomap follows the Style Guide for Python Code. These are just some rules for consistency that you can read all about in the `Python Style Guide <https://www.python.org/dev/peps/pep-0008/>`_.
3. Write tests
4. Write documentation.
5. The Python programming language, `for beginners <https://www.python.org/about/gettingstarted/>`_.
