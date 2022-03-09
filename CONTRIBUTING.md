-----

<h3 align="center">We are continuously working towards our primary goal of accessible asteroseismology</h3>

-----


## Community Guidelines

For most (if not all) questions/concerns, peeping our [discussions](https://github.com/ashleychontos/pySYD/discussions) forum is an excellent place to start in case things have already been brought to our attention and/or addressed.

As a contributor, here are the guidelines we ask you to follow:

* [Question or problem](#question)
* [Issues & bugs](#issue)
* [New features](#feature)
* [Contributing code](#contribute)
* [Style guide](#style)
* [Testing](#testing)

<a name="question"></a>
## Question or problem

&rightarrow; Do you have a general question that is not directly related to software functionality?

Please visit our relevant [thread](https://github.com/ashleychontos/pySYD/discussions/37#discussion-391811) first to see if your question has already been asked. You can also help us keep this space up-to-date, linking topics/issues to relevant threads and adding appropriate tags whenever/wherever possible. This is not only helpful to us but also helpful for the community! Once we have enough data points, we will establish a forum for frequently asked questions (FAQ).

---
**NOTE**

**Please do not open an issue for general support questions as we want to preserve them for bug reports
and new feature requests ONLY. Therefore to save everyone time, we will be systematically closing all 
issues that do not follow these guidelines.**

---

If this still does not work for you and you would like to chat with someone in real-time, please contact [Ashley](mailto:achontos@hawaii.edu) to set up a chat or zoom meeting.

<a name="issue"></a>
## Issues & bugs

&rightarrow; Are you reporting a bug?

**If the code crashes or you find a bug, please search the issue tracker first to make sure the problem (i.e. issue) does not already exist.** If and only if you do this but still don't find anything, feel free to submit an issue. And, if you're *really* brave, you can submit an issue along with a pull request fix.

Ideally we would love to resolve all issues immediately but before fixing a bug, we first to need reproduce and confirm it. There is a template of the required information when you submit an issue, but generally we ask that:

* clearly and concisely explain the issue or bug
* provide the light curve and power spectrum data so that we can reproduce it from our end
* what operating system you ran the software on and what version of the software you're using

You can file new issues by filling out our [bug report](https://github.com/ashleychontos/pySYD/issues/new?assignees=&labels=&template=bug_report.md) template.

<a name="feature"></a>
## New features

&rightarrow; Have an idea for a new feature or functionality?


### Indirect request 

If you come up with an idea for a new feature that you'd like to see implemented in 
``pySYD`` but do not plan to do this yourself, you can submit an issue with our 
[feature request](https://github.com/ashleychontos/pySYD/issues/new?assignees=&labels=&template=feature_request.md) template.

We welcome any and all ideas!


### Directly implement

However, if you come up with a brilliant idea that you'd like to take a stab at -- 
please first consider what kind of change it is:

* For a **Major Feature**, first 
  [open an issue and outline your proposal](https://github.com/ashleychontos/pySYD/issues/new?assignees=&labels=&template=outline_major_pr.md) 
  so that it can be discussed. This will also allow us to better coordinate our efforts, prevent duplication of work,
  and help you to craft the change so that it is successfully accepted into the project.
* Any smaller or **Minor Features** can be crafted and directly submitted as a pull request. However,
  before you submit a pull request, please see our [style guide](#style) to facilitate and expedite 
  the merge process.

<a name="contribute"></a>
## Contributing code

&rightarrow; Do you want to contribute code?

We would love for you to contribute to `pySYD` and make it even better than it is today! 

### Submitting a Pull Request (PR)

**PR reminders:**
 - PR title and description should be as clear as possible
 - please follow our [guide](#style) for code and docstrings formats
 - link back to the original issue(s) whenever possible
 - large pull requests should be broken into separate pull requests (or multiple logically cohesive commits), if possible
 - new commands should be added to `docs/support_table.md` and `docs/supported.md`

<a name="style"></a>
## Style guide

&rightarrow; A good rule of thumb is to try to make your code blend in with the surrounding code

### Code
 * 4 spaces for indentation (i.e. no tabs please)
 * 80 character line length
 * commas last
 * declare variables in the outermost scope that they are used
 * camelCase for classes/objects in Python
 * snake_case for variables in Python

### Docstrings
 * compliant with PEP 257
 * numpydoc

### Coding Rules

To ensure consistency throughout the source code, keep these rules in mind as you are working:

* All features or bug fixes **must be tested** by one or more specs (unit-tests).
* We follow [Google's JavaScript Style Guide][js-style-guide].

<a name="testing"></a>
## Testing
