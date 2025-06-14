from onedl.client import connect_to_project

client = connect_to_project('daniel-osman---streamlining-annotation-bootstrapping/testing')

past = client.datasets.load('train-f0:1')
new_version_name = client.datasets.save('train-f0', past, exist="versions")
client.datasets.push(new_version_name, push_policy="version")
